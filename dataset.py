import pickle
from pathlib import Path

import torch
import torchaudio

import pandas as pd


def get_mel_spectrogram(file_path, save_dir, SR = 22050):
    file_name = file_path.split('/')[-1].replace('.mp3','')
    audio, sr = torchaudio.load(file_path)
    audio = audio.mean(dim=0)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)(audio)
    
    spec = torchaudio.functional.spectrogram(y, n_fft=1024,
                                         hop_length=512, 
                                         pad=0, 
                                         win_length=1024,
                                         power=True,
                                         normalized=True,
                                         window = torch.hann_window(1024)
                                        )
    
    mel_scale = torchaudio.transforms.MelScale(n_mels=128, 
                                           sample_rate=22050, 
                                           f_min=20, 
                                           f_max=8000, 
                                           n_stft=spec.shape[0])
    mel_spec = mel_scale(spec)
    
    mel_spec = torch.log(1+10 * mel_spec).unsqueeze(-1)
    
    for i in range(4):
        torch.save(mel_spec[:,107*i:107*(i+1)], f"{save_dir}/{file_name}_{i}.pt")

def preprocess(root_dir, aud_file, save_dir, SR = 22050):
    files = pickle.load(open(aud_file, 'rb'))
    for file in files:
        aud_path = root_dir + '/' + file
        get_mel_spectrogram(aud_path, save_dir, SR)


class OurDataset:
    
    def __init__(self, root_dir, aud_file, label_file, SR = 22050):
        self.root = root_dir
        self.files = pickle.load(open(aud_file, 'rb'))
        self.labels = pickle.load(open(label_file, 'rb'))
        self.sr = SR
        self.threshold = 2
        
    def convert_label_to_tensor(self):
        return torch.LongTensor(self.train.values[:, 1:-1].astype('bool'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        aud_path = self.root + '/' + self.files[idx]
        audio, sr = torchaudio.load(aud_path)
        audio = audio.mean(dim=0)
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(audio)
        
        label_dict = self.labels[idx]
        label = torch.tensor(list(label_dict.values))
        label = (label > self.threshold).to(int)
        
        return audio, label  # mel_spectrogram, label

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