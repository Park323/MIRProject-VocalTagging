import pdb
import os
import argparse
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
    print(1)
    ## Load Data
    VAL_DIR = args.valid_dir
    TEST_DIR = args.test_dir
    
    entire_label = ['Sad', 'Thick', 'Warm', 'Clear', 'Dynamic', 'Energetic', 'Speech-Like', 'Sharp', 'Falsetto', 'Robotic/Artificial', 
                     'Whisper/Quiet', 'Delicate', 'Passion', 'Emotional', 'Mid-Range', 
                     'High-Range', 'Compressed', 'Sweet', 'Soulful/R&B', 'Stable', 
                     'Rounded', 'Thin', 'Mild/Soft', 'Breathy', 'Pretty', 
                     'Young', 'Dark', 'Husky/Throaty', 'Bright', 'Vibrato', 
                     'Pure', 'Male', ' Ballad', 'Rich', 'Low-Range', 
                     'Shouty', 'Cute', 'Relaxed', 'Female', 'Charismatic', 
                     'Lonely', 'Embellishing']
                     
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
    # Technique & Low-level Timbre
    sets['B'] = ['Male', 'Female', 'Whisper/Quiet', 'Shouty', 'Vibrato', 
                 'Falsetto', 'Speech-like', 'Compressed', 'Husky/Throaty', 'Thick', 
                 'Thin', 'Sharp', 'Breathy', 'Stable']
    # High level Timbre
    sets['C'] = ['Lonely', 'Sad', 'Passion','Charismatic','Pretty',
                'Cute','Delicate','Emotional','Pure','Robotics/Artificial',
                'Embellishing','Sweet','Young','Compressed','Dynamic']
    
    print(2)
    label_filter = [(label in sets[args.label_filter]) for label in entire_label]
    label_filter = torch.tensor(label_filter)
    valset = OurDataset(VAL_DIR, label_filter)
    valloader = DataLoader(valset, batch_size=args.batch_size)
    testset = OurDataset(TEST_DIR, label_filter)
    testloader = DataLoader(testset, batch_size=args.batch_size)
    print(3)
    n_class = int(label_filter.sum())
    
    ## Define Model
    model = get_model(args.model, n_class).to(DEVICE)
    model.load_state_dict(torch.load(args.pt))
    print(4)
    
    criterion = get_criterion(args.loss)
    
    threshold=torch.zeros((n_class,)).to(DEVICE)
    
    unit = 0.6 if args.loss=='mse' else 0.2
    max_scores = []
    model.eval()
    for thres_idx in tqdm(range(n_class)):
        maximum_score = 0
        maximum_threshold = 0
        for i in range(1,5):
            metric = get_metric(torch.tensor([unit*i]).to(DEVICE))
            total_score = 0
            for data, label in valloader:
                data = data.to(DEVICE)
                label = label.to(float).to(DEVICE)
                
                output = model(data)
                
                score = metric(output[:,thres_idx:thres_idx+1], label[:,thres_idx:thres_idx+1])
                total_score += score.detach().item()
            mean_score = total_score/len(valloader)
            if mean_score > maximum_score:
                maximum_threshold = unit*i
                maximum_score = mean_score
        threshold[thres_idx] = maximum_threshold
        print(maximum_threshold, maximum_score)
        max_scores.append(maximum_score)
    
    metric = get_metric(threshold=threshold)
    
    ## Test
    loss, score = test(model, valloader, criterion, metric)
    print(loss, score)
    
    loss, score = test(model, testloader, criterion, metric)
    print(loss, score)
    
    print(max_scores)
    print(threshold)
    
    torch.save(threshold, f"{(args.pt).replace('.pt','_threshold.pt')}")
    

def get_args():
    parser = argparse.ArgumentParser(description="configure train")
    parser.add_argument('--valid_dir', default='data')
    parser.add_argument('--test_dir', default='data')
    parser.add_argument('--pt')
    parser.add_argument('--loss', default='bce')
    parser.add_argument('--model', default='base')
    parser.add_argument('--label_filter', '-lb', default='X')
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    main(args)