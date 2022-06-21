import torch
import torch.nn as nn
#from torchvision.models import resnet50


class base_model(nn.Module):
    '''
    Base model from "Semantic Tagging of Singing Voices in Popular Music Recordings"
    input : spectrogram
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self, output_dim):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
        )
        self.fc = nn.Linear(5*4*128, output_dim)
    
    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc(output.view(output.size(0),-1))
        return output
    
    
class drop_model(nn.Module):
    '''
    Base model from "Semantic Tagging of Singing Voices in Popular Music Recordings"
    input : spectrogram
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self, output_dim):
        super().__init__()
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Dropout2d(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(3, 3, ceil_mode=True),
            nn.Dropout2d(),
        )
        self.fc = nn.Linear(5*4*128, output_dim)
    
    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc(output.view(output.size(0),-1))
        return output


class ResidualCell2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, identity=True):
        super().__init__()
        if identity:
            self.shortcut = nn.Identity(stride=2)
            stride = 1
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=3)
            stride = 3
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size, stride=stride, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        Fx = self.conv(x)
        x = self.shortcut(x)
        return Fx + x


class resnet_model(nn.Module):
    '''
    Base model from "Semantic Tagging of Singing Voices in Popular Music Recordings"
    input : spectrogram
    output : predicted tag distribution (one-hot encoded)
    '''
    def __init__(self, output_dim):
        super().__init__()
        self.conv_layers=nn.Sequential(
            ResidualCell2d(1, 32, 3),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(32, 32, 3),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(32, 64, 3, False),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(64, 128, 3, False),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(128, 128, 3, False),
            nn.ReLU(),
            nn.Dropout2d(),
        )
        self.fc = nn.Linear(5*4*128, output_dim)
    
    def forward(self, x):
        output = self.conv_layers(x)
        output = self.fc(output.view(output.size(0),-1))
        return output


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers=nn.Sequential(
            ResidualCell2d(1, 32, 3),
            nn.ReLU(),
            ResidualCell2d(32, 32, 3),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(32, 64, 3, False),
            nn.ReLU(),
            ResidualCell2d(64, 64, 3),
            nn.ReLU(),
            nn.Dropout2d(),
            ResidualCell2d(64, 128, 3, False),
            nn.ReLU(),
            ResidualCell2d(128, 128, 3),
            nn.ReLU(),
            nn.Dropout2d(),
        ) # (B, 128, 5, 4)
        self.rnn = None
        
    def forward(self, x):
        return
    
# class resnet2d_model(nn.Module):
#     '''
#     Use ResNet50 which is pretrained with ImageNet
#     input : spectrogram (B,1,H,W)
#     output : predicted tag distribution (one-hot encoded)
#     '''
#     def __init__(self):
#        super().__init__()
#        self.resnet = resnet50(pretrained=True)
#        self.fc = nn.Linear(1000, 42)
    
#     def forward(self, x):
#        x = x.repeat(1, 3, 1, 1)
#        output = self.resnet(x)
#        output = self.fc(output)
#        return output