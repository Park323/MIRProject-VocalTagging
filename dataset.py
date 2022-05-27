import torch
import pandas as pd

class SampleDataset:
    def __init__(self, csv_path, split='train', sr=16000):
        pass
        
    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        # return (data, label)
        # C x L x F
        label = torch.zeros(42)
        label[0] = 1
        return torch.rand(1,128,107), label