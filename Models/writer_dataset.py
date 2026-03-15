import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import random
import os
from typing import Sequence, List, Optional, Dict, Tuple
from load_data import IMAGE_HEIGHT,IMAGE_WIDTH,CHAR_TO_IDX,MAX_L,K,ALPHABET_SIZE


class IAMDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.writer_groups = self.df.groupby('Writer_ID')
        self.writers = list(self.writer_groups.groups.keys())
        self.writer_to_idx = {w: i for i, w in enumerate(self.writers)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        writer = random.choice(self.writers)
        group = self.writer_groups.get_group(writer)
        samples = group.sample(K+1, replace=len(group) < K+1)
        Xi_paths = samples.iloc[:K]['Filepath'].tolist()
        x_path = samples.iloc[-1]['Filepath']
        t_str = samples.iloc[-1]['Text']
        Xi = torch.stack([self.transform(Image.open(path)) for path in Xi_paths])  # (K, 1, 64, 256)
        x = self.transform(Image.open(x_path))  # (1, 64, 256)
        t = torch.zeros(MAX_L, ALPHABET_SIZE)
        if not isinstance(t_str, str) or pd.isna(t_str):
            t_str = 'ε' * MAX_L
        len_str = min(len(t_str), MAX_L)
        for i in range(len_str):
            char = t_str[i]
            if char in CHAR_TO_IDX:
                t[i, CHAR_TO_IDX[char]] = 1
        t[len_str:, CHAR_TO_IDX['ε']] = 1
        return Xi.squeeze(1), x, t, self.writer_to_idx[writer]