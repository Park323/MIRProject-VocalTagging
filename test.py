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


def test(model, dataloader, criterion, metric):
    '''
    Test model with data
    
    (input)
    model : model to train
    dataloader : dataloader
    criterion : loss function
    optimizer : optimizer
    history : history dictionary for record loss & metrics
    
    (output)
    None
    '''
    model.eval()
    
    batch_idx = 1
    mean_loss = 0.
    mean_score = 0.

    with torch.no_grad():        
        process = tqdm(dataloader)
        for data, label in process:
            data = data.to(DEVICE)
            label = label.to(float).to(DEVICE)
            
            output = model(data)
            
            # output : (B, C)
            # label : (B, C)
            loss = criterion(output, label)
            score = metric(output, label)
            
            mean_loss += loss.detach().item()
            mean_score += score.detach().item()
            
            process.set_description(f"LOSS  : {loss:.4f}, MEAN LOSS : {mean_loss/batch_idx:.4f}, F-score  : {score:.4f}, MEAN score : {mean_score/batch_idx:.4f}")
            
            batch_idx += 1
        
    return mean_loss/batch_idx, mean_score/batch_idx

def main(args):

    ## Load Data
    TEST_DIR = args.test_dir
    
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
    testset = OurDataset(TEST_DIR, label_filter)
    testloader = DataLoader(testset, batch_size=args.batch_size)
    
    ## Define Model
    model = get_model(args.model, len(label_filter)).to(DEVICE)
    model.load_state_dict(torch.load(args.pt))
    
    ## Define functions for training
    threshold=torch.full((len(label_filter),), 0.5)
    
    criterion = get_criterion()
    metric = get_metric(threshold=threshold)
    
    model.eval()
    for thres_idx in tqdm(range(len(label_filter))):
        maximum_score = 0
        maximum_threshold = 0
        for i in range(1,5):
            metric = get_metric(torch.tensor([0.2*i]))
            total_score = 0
            for data, label in testloader:
                data = data.to(DEVICE)
                label = label.to(float).to(DEVICE)
                
                output = model(data)
                
                score = metric(output[:,i:i+1], label[:,i:i+1])
                total_score += score.detach().item()
            mean_score = total_score/len(testloader)
            if mean_score > maximum_score:
                maximum_threshold = 0.1*i
                maximum_score = mean_score
        threshold[thres_idx] = maximum_threshold
        print(maximum_threshold, maximum_score)
    
    metric = get_metric(threshold)
    ## Test
    loss, score = test(model, testloader, criterion, metric)
    print(loss, score)
    

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    parser.add_argument('--test_dir', default='data')
    parser.add_argument('--pt')
    parser.add_argument('--model', default='base')
    parser.add_argument('--label_filter', '-lb', default='X')
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)