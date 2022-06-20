import torch
import torch.nn as nn
from torchvision.models import resnet50


class base_model(nn.Module):
    '''
    Base model from "Semantic Tagging of Singing Voices in Popular Music Recordings"
    input : spectrogram
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
        )
        self.fc = nn.Linear(5*4*128, 42)
    
    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc(output.view(output.size(0),-1))
        return output
    

class resnet2d_model(nn.Module):
    '''
    Use ResNet50 which is pretrained with ImageNet
    input : spectrogram (B,1,H,W)
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self):
       super().__init__()
       self.resnet = resnet50(pretrained=True)
       self.fc = nn.Linear(1000, 42)
    
    def forward(self, x):
       x = x.repeat(1, 3, 1, 1)
       output = self.resnet(x)
       output = self.fc(output)
       return output