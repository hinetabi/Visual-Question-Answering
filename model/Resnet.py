import torch
import torchvision.models as models
import torchvision

def get_pretrained_resnet():
    
    return 0


class PretrainedObjectDetectionModel():
    def __init__(self, pretrained = False):
        self.pretrain = pretrained
        
        
    def get_model(self, model_name):
        
        if model_name == 'resnet':
            model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True, pretrained_backbone=True)
        elif model_name == '':
            # model = models.detection.
            model = 0
        
        return model

# import torch.optim as optim

# optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2)