import torch
import torch.nn as nn


def get_criterion():
    return nn.BCEWithLogitsLoss()

    
def get_optimizer(model, lr, method='adam'):
    if method == 'adam':
        return torch.optim.Adam(model.parameters(), lr)
    

def get_scheduler(optimizer, method='none'):
    if method == 'none':
        return None