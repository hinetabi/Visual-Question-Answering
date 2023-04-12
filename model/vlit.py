from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertConfig, BertModel, BertForQuestionAnswering
from model.ViT import VisionTransformer
from functools import partial
from transformers import AutoModelForSequenceClassification

class QuestionAnswerClassifier(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None, 
                 n_labels = 3129    
                 ):
        super().__init__()
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=4, num_heads=2, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))  
        # self.image_encoder = 
        
        # create the text encoder
        config_encoder = BertConfig.from_json_file(config['bert_config']) 
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
        
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layer = 6
        # self.text_decoder = BertForQuestionAnswering.from_pretrained(text_decoder, config = config_decoder)
        
        # self.tokenizer =  tokenizer
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=768*2),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(in_features=768*2, out_features=n_labels),
            nn.Dropout(p=0.1, inplace=False)
        )        
    def forward(self, image, question):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        
        # encode the question (with the image embeds)
        question_encoder =  self.text_encoder(
                                            question.input_ids, 
                                            attention_mask = question.attention_mask, 
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,                             
                                            return_dict = True  )
        
        
        # process the question_encoder after a classification
        # cls token is the first of last hidden state
        return self.classifier(question_encoder.last_hidden_state[:, 0, :])
