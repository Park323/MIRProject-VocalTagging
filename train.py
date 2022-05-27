import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SampleDataset
from model import base_model
from utils import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, dataloader, criterion, optimizer, history, train=True):
    '''
    Train model with data
    
    (input)
    model : model to train
    dataloader : dataloader
    criterion : loss function
    optimizer : optimizer
    history : history dictionary for record loss & metrics
    train : check the process is for train or validation
    
    (output)
    None
    '''
    if train: model.train()
    else: model.eval()
    
    process = tqdm(dataloader)
    for data, label in process:
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = criterion(output, label)
        
        loss.backward()
        optimizer.step()
        
        if train:
            history['train_loss'].append(loss.detach().item())
        else:
            history['valid_loss'].append(loss.detach().item())
            
        process.set_description(f"LOSS : {loss:.2f}")


def main(args):
    ## Load Data
    DATA_PATH = args.data_path
    trainset = SampleDataset(DATA_PATH)
    trainloader = DataLoader(trainset, batch_size=8)
    
    ## Define Model
    model = base_model().to(DEVICE)
    
    ## Define functions for training
    criterion = get_criterion()
    optimizer = get_optimizer(model, args.learning_rate)
    
    ## Train
    history = {'train_loss':[], 
               'valid_loss':[]}
    for _ in tqdm(range(args.epoch)):
        train(model, trainloader, criterion, optimizer, history, train=True)
        train(model, trainloader, criterion, optimizer, history, train=False)
    

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--model', default='base')
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)