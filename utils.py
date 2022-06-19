import torch
import torch.nn as nn


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
    
    
class F_score(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        '''
        x : (N, 42)
        '''
        return None