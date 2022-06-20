import torch
import torch.nn as nn

from model import base_model#, resnet2d_model

def get_criterion():
    return nn.BCEWithLogitsLoss()

def get_metric(threshold=None):
    return F_score(threshold)
    
def get_optimizer(model, lr, method='adam'):
    if method == 'adam':
        return torch.optim.Adam(model.parameters(), lr)
    

def get_scheduler(optimizer, method='none'):
    if method == 'none':
        return None
        
        
def get_model(model, output_dim=None):
    if model=='base':
        return base_model(output_dim)
    elif model=='resnet2d':
        return resnet2d_model()
    
    
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
        x : (N, 42)
        labels : (N, 42)
        '''
        predict = (x>self.THRESHOLDS)
        tp = labels[predict].sum()
        retrieved = predict.sum()
        relevant = labels.sum()
        precision = tp/(retrieved+1e-10)
        recall = tp/relevant
        f_score = 2*(precision*recall)/(precision+recall+1e-10)
        
        return f_score
