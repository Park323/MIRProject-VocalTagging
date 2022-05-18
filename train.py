import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import base_model

def train(model, dataloader, criterion, optimizer):
    '''
    Train model with data
    
    (input)
    model : model to train
    dataloader : dataloader
    criterion : loss function
    optimizer : optimizer
    
    (output)
    None
    '''
    pass

def main(args):
    pass

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)