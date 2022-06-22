import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import base_model, drop_model, resnet_model, CRNN, SimpleCRNN

def get_criterion(method='bce', pos_weight=None):
    if method=='bce':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif method=='mse':
        mse = nn.MSELoss()
        return lambda x, y: mse(x.to(float), y)

def get_metric(threshold=None):
    return F_score(threshold=threshold)
    
def get_optimizer(model, lr, method='adam'):
    if method == 'adam':
        return torch.optim.Adam(model.parameters(), lr)
    

def get_scheduler(optimizer, method='none'):
    if method == 'none':
        return None
        
        
def get_model(model, output_dim=None):
    if model=='base':
        return base_model(output_dim)
    elif model=='drop':
        return drop_model(output_dim)
    elif model=='resnet':
        return resnet_model(output_dim)
    elif model=='crnn':
        return CRNN(output_dim)
    elif model=='simple':
        return SimpleCRNN(output_dim)
    
    
class F_score(nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        if threshold is not None:
            self.THRESHOLDS = threshold
        else:
            self.THRESHOLDS = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
                                            0.5, 0.5])
        
    def forward(self, x, labels, train=False):
        '''
        x : (N, C)
        labels : (N, C)
        '''
        #pdb.set_trace()
        predict = (x>self.THRESHOLDS)
        tp = ((labels==1)*(predict==1)).sum()
        retrieved = predict.sum()
        relevant = labels.sum()
        precision = tp/(retrieved+1e-10)
        recall = tp/(relevant+1e-10)
        
        f_score = 2*(precision*recall)/(precision+recall+1e-10)
        
        return f_score, precision, recall
