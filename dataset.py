import os
import glob
import tqdm
import pickle
from pathlib import Path

import torch
import torchaudio

import pandas as pd


def get_mel_spectrogram(file_path, SR = 22050):
    audio, sr = torchaudio.load(file_path)
    audio = audio.mean(dim=0)
    audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SR)(audio)
    
    spec = torchaudio.functional.spectrogram(audio, n_fft=1024,
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
    
    mel_spec = torch.log(1+10 * mel_spec).unsqueeze(0)
    
    return mel_spec
    
def preprocess(root_dir, aud_file, label_file, save_dir, threshold = 2, SR = 22050):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    files = pickle.load(open(aud_file, 'rb'))
    labels = pickle.load(open(label_file, 'rb'))
    
    for file, label in zip(files, labels):
        aud_path = root_dir + '/' + file
        mel_spec = get_mel_spectrogram(aud_path, SR)
        
        label = torch.tensor(list(label.values()))
        label = (label >= threshold).to(int)
        
        for i in range(4):
            data = (mel_spec[:,:,107*i:107*(i+1)], label)
            torch.save(data, f"{save_dir}/{file}_{i}.pt")
            
            
def preprocess2(root_dir, aud_file, label_file, save_dir, threshold = 2, SR = 22050):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pts = os.listdir(root_dir)
    
    files = pickle.load(open(aud_file, 'rb'))
    labels = pickle.load(open(label_file, 'rb'))
    
    file2label = {file:label for file, label in zip(files, labels)}
    
    for pt in tqdm.tqdm(pts):
        path = root_dir + '/' + pt
        y, l = torch.load(open(path, 'rb'))
        pt = pt[:-5]
        label = file2label[pt]
        
        for i in range(4):
            data = (y, label)
            torch.save(data, f"{save_dir}/{pt}_{i}.pt")


class OurDataset:
    label = ['Sad', 'Thick', 'Warm', 'Clear', 'Dynamic', 'Energetic', 'Speech-Like', 'Sharp', 'Falsetto', 'Robotic/Artificial', 
             'Whisper/Quiet', 'Delicate', 'Passion', 'Emotional', 'Mid-Range', 
             'High-Range', 'Compressed', 'Sweet', 'Soulful/R&B', 'Stable', 
             'Rounded', 'Thin', 'Mild/Soft', 'Breathy', 'Pretty', 
             'Young', 'Dark', 'Husky/Throaty', 'Bright', 'Vibrato', 
             'Pure', 'Male', ' Ballad', 'Rich', 'Low-Range', 
             'Shouty', 'Cute', 'Relaxed', 'Female', 'Charismatic', 
             'Lonely', 'Embellishing']
    
    def __init__(self, root_dir, chosen_labels):
        self.files = glob.glob(f"{root_dir}/*.pt")
        self.chosen_labels = chosen_labels
        self.filter = [(label in self.chosen_labels) for label in OurDataset.label]
        self.filter = torch.tensor(self.filter)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio, label = torch.load(self.files[idx])
        label = label[self.filter]
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