import torch
import torch.nn as nn

from model import base_model, resnet2d_model

def get_criterion():
    return nn.BCELoss()

def get_metric():
    return None
    
def get_optimizer(model, lr, method='adam'):
    if method == 'adam':
        return torch.optim.Adam(model.parameters(), lr)
    

def get_scheduler(optimizer, method='none'):
    if method == 'none':
        return None
        
        
def get_model(model):
    if model=='base':
        return base_model()
    elif model=='resnet2d':
        return resnet2d_model()
    
    
class F_score(nn.Module):
    def __init__(self):
        super().__init__()
        self.THRESHOLDS = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                        0.5, 0.5])
        
    def forward(self, x):
        '''
        x : (N, 42)
        '''
        return None
