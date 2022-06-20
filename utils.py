import torch
import torch.nn as nn

from model import base_model, resnet2d_model

def get_criterion():
    return nn.BCEWithLogitsLoss()

    
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
        