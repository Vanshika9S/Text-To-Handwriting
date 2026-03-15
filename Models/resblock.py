from typing import List, Sequence, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from adain import adain

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, alpha1, beta1, alpha2, beta2):
        residual = x
        x = adain(self.relu(self.conv1(x)), alpha1, beta1)
        x = adain(self.conv2(x), alpha2, beta2)
        return x + residual
