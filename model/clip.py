from torchvision.models import resnet50, ResNet50_Weights
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertConfig, BertModel, BertForQuestionAnswering


class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        self.image_encoder = resnet50(ResNet50_Weights.IMAGENET1K_V2)
        # self.image_encoder = 
        
        # create the text encoder
        config_encoder = BertConfig.from_json_file(config['bert_config']) 
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=config_encoder, add_pooling_layer=False)  
        
        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layer = 6
        self.text_decoder = BertForQuestionAnswering.from_pretrained(text_decoder, config = config_decoder)
        

    def forward(self, image, question, answer=None, alpha=0, k=None, weights=None, train=True):
        image_encoded = self.image_encoder(image)
        


if __name__ == '__main__':
    image_encoder = resnet50(ResNet50_Weights.IMAGENET1K_V2)
    print(image_encoder)