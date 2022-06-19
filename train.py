import pdb
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import SampleDataset, OurDataset
from model import base_model
from utils import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, dataloader, criterion, optimizer, history, epoch=0, train=True):
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
    
    batch_idx = 1
    mean_loss = 0.
    
    process = tqdm(dataloader)
    for data, label in process:
        data = data.to(DEVICE)
        label = label.to(float).to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        loss = criterion(output, label)
        
        if train:
            loss.backward()
            optimizer.step()
            
            history['train_loss'].append(loss.detach().item())
        else:
            history['valid_loss'].append(loss.detach().item())
        
        mean_loss += loss.item()
        
        process.set_description(f"EPOCH : {epoch}, LOSS  : {loss:.4f}, MEAN LOSS : {mean_loss/batch_idx:.4f}")
        
        batch_idx += 1

def main(args):

    ## Load Data
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.valid_dir
    
    trainset = OurDataset(TRAIN_DIR)
    valset = OurDataset(VAL_DIR)
    trainloader = DataLoader(trainset, batch_size=8)
    valloader = DataLoader(valset, batch_size=8)
    
    ## Define Model
    model = base_model().to(DEVICE)
    
    ## Define functions for training
    criterion = get_criterion()
    optimizer = get_optimizer(model, args.learning_rate)
    
    ## Train
    history = {'train_loss':[], 
               'valid_loss':[]}
    for epoch in range(args.epoch):
        train(model, trainloader, criterion, optimizer, history, epoch=epoch, train=True)
        train(model, valloader, criterion, optimizer, history, epoch=epoch, train=False)
        
        if not os.path.exists('results'):
            os.makedirs('results')
        torch.save(model.state_dict(), f"results/model_{epoch}.pt")
            
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.savefig("results/tr_graph.png")
        
    

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    parser.add_argument('--train_dir', default='data')
    parser.add_argument('--valid_dir', default='data')
    parser.add_argument('--model', default='base')
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)