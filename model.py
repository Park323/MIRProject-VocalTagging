import torch
import torch.nn as nn

class base_model(nn.Module):
    '''
    Base model from "Semantic Tagging of Singing Voices in Popular Music Recordings"
    input : spectrogram
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self):
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(32, 64, 3, 1),
            nn.MaxPool2d(3, 3),
            nn.Conv2d(64, 128, 3, 1),
            nn.MaxPool2d(3, 3),
        )
        self.fc = nn.Linear(5*4*128, 42)
    
    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc(output.flatten())
        return output