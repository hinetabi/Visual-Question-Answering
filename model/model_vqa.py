from functools import partial
from model.ViT import VisionTransformer
from transformers import BertConfig, BertModel, BertLMHeadModel

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,  
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=4, num_heads=2, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))  

        config_encoder = BertConfig.from_json_file(config['bert_config']) 
        # create the text encoder
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
        
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    

        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        

    def forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True, generate = False, tokenizer = None):
        
        # encode the image
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)      

            # encode the question (with the image embeds)
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = True)    

            
            # information of question, includes question_output last hidden state and question attention mask.
            question_states = []                
            question_atts = []  
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0)     

            # decode the answer based on questions state and questions att (output of multimodal encoder)
            answer_output = self.text_decoder(answer.input_ids, 
                                                attention_mask = answer.attention_mask, 
                                                encoder_hidden_states = question_states,
                                                encoder_attention_mask = question_atts,                  
                                                labels = answer_targets,
                                                return_dict = True   
                                            #   reduction = 'none',
                                            )                      
            loss = weights * answer_output.loss         
            loss = loss.sum()/image.size(0)

            return loss
        if generate: # generate # if generate, question is not tokenized before load into model
            question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
            
            question.input_ids[:,0] = self.tokenizer.enc_token_id

            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True) 
            
            # generate 
            num_beams = 3

            question_states = []                
            question_atts = []  
            # k = [3]

            # add state and att for cross attention
            for b, n in enumerate([num_beams]):
                question_states += [question_output.last_hidden_state[b]]*n
                question_atts += [question.attention_mask[b]]*n 
            question_states = torch.stack(question_states,0)    
            question_atts = torch.stack(question_atts,0) 

            model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}
            
            bos_ids = torch.full((image.size(0),1),fill_value=self.tokenizer.bos_token_id,device=image.device)
            
            outputs = self.text_decoder.generate(input_ids=bos_ids,
                                                    max_length=10,
                                                    min_length=1,
                                                    num_beams=num_beams,
                                                    eos_token_id=self.tokenizer.sep_token_id,
                                                    pad_token_id=self.tokenizer.pad_token_id, 
                                                    **model_kwargs)
            
            answers = []    
            for output in outputs:
                answer = self.tokenizer.decode(output, skip_special_tokens=True)    
                answers.append(answer)
            return answers
        
        if not train: 
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)
            
                                
            topk_ids, topk_probs = self.rank_answer(question_output.last_hidden_state, question.attention_mask, 
                                                    answer.input_ids, answer.attention_mask, k) 
            return topk_ids, topk_probs

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True)
                                        #  reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = [prob_first_token.topk(k,dim=1) for k in range(k) ]
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True) 
                                #    reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
