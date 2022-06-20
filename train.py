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


def train(model, dataloader, criterion, metric, optimizer, history, epoch=0, train=True):
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
    mean_score = 0.
    
    process = tqdm(dataloader)
    for data, label in process:
        data = data.to(DEVICE)
        label = label.to(float).to(DEVICE)
        
        optimizer.zero_grad()
        
        output = model(data)
        
        # output : (B, C)
        # label : (B, C)
        loss = criterion(output, label)
        with torch.no_grad():
            score = metric(output, label)
        
        if train:
            loss.backward()
            optimizer.step()
        
        mean_loss += loss.detach().item()
        mean_score += score.detach().item()
        
        process.set_description(f"EPOCH : {epoch}, LOSS  : {loss:.4f}, MEAN LOSS : {mean_loss/batch_idx:.4f}, F-score  : {score:.4f}, MEAN score : {mean_score/batch_idx:.4f}")
        
        batch_idx += 1
        
    return mean_loss, mean_score

def main(args):

    ## Load Data
    TRAIN_DIR = args.train_dir
    VAL_DIR = args.valid_dir
    
    sets = {}
    sets['X'] = ['Sad', 'Thick', 'Warm', 'Clear', 'Dynamic', 'Energetic', 'Speech-Like', 'Sharp', 'Falsetto', 'Robotic/Artificial', 
                'Whisper/Quiet', 'Delicate', 'Passion', 'Emotional', 'Mid-Range', 
                'High-Range', 'Compressed', 'Sweet', 'Soulful/R&B', 'Stable', 
                'Rounded', 'Thin', 'Mild/Soft', 'Breathy', 'Pretty', 
                'Young', 'Dark', 'Husky/Throaty', 'Bright', 'Vibrato', 
                'Pure', 'Male', ' Ballad', 'Rich', 'Low-Range', 
                'Shouty', 'Cute', 'Relaxed', 'Female', 'Charismatic', 
                'Lonely', 'Embellishing']
    sets['A'] = ['Rounded', 'Pretty', 'Delicate', 'Sharp', 'Passion', 
                               'Lonely', 'Compressed', 'Pure', 'Sweet', 'Husky/Throaty', 
                               'Rich', 'Energetic', 'Young', 'Robotic/Artificial', 'Clear', 
                               'Thin', 'Thick', 'Mild/Soft', 'Bright', 'Charismatic',
                               'Embellishing', 'Breathy', 'Dynamic', 'Cute', 'Sad',
                               'Stable', 'Emotional', 'Warm', 'Relaxed', 'Dark']
    
    label_filter = sets[args.label_filter]
    trainset = OurDataset(TRAIN_DIR, label_filter)
    valset = OurDataset(VAL_DIR, label_filter)
    trainloader = DataLoader(trainset, batch_size=args.batch_size)
    valloader = DataLoader(valset, batch_size=args.batch_size)
    
    ## Define Model
    model = get_model(args.model, len(label_filter)).to(DEVICE)
    
    ## Define functions for training
    threshold=torch.full((len(label_filter),), 0.5)
    criterion = get_criterion(args.loss)
    if args.loss=='mse':
        metric = get_metric(threshold=threshold, ltype='level')
    else:
        metric = get_metric(threshold=threshold)
    optimizer = get_optimizer(model, args.learning_rate)
    
    ## Train
    history = {'train_loss':[],
               'train_score':[],
               'valid_loss':[],
               'valid_score':[]}
    for epoch in range(args.epoch):
        print("Train")
        tr_loss, tr_score = train(model, trainloader, criterion, metric, optimizer, history, epoch=epoch, train=True)
        print("Valid")
        vl_loss, vl_score = train(model, valloader, criterion, metric, optimizer, history, epoch=epoch, train=False)
        
        history['train_loss'].append(tr_loss)
        history['train_score'].append(tr_score)
        history['valid_loss'].append(vl_loss)
        history['valid_score'].append(vl_score)
        
        if not os.path.exists('results'):
            os.makedirs('results')
        torch.save(model.state_dict(), f"results/model_{epoch}.pt")
        
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.savefig("results/loss.png")
    
    plt.plot(history['train_score'])
    plt.plot(history['valid_score'])
    plt.savefig("results/score.png")
    
    
    

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    parser.add_argument('--train_dir', default='data')
    parser.add_argument('--valid_dir', default='data')
    parser.add_argument('--model', default='base')
    parser.add_argument('--loss', default='bce')
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--label_filter', '-lb', default='X')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)