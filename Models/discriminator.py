import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class DiscriminatorResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.pool = nn.AvgPool2d(2)
        self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1), nn.AvgPool2d(2))

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.pool(self.leaky(self.conv2(self.leaky(self.conv1(x)))))
        return x + residual
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        channels = [64, 128, 256, 512, 512, 512]
        self.res_blocks = nn.ModuleList([DiscriminatorResBlock(channels[i], channels[i+1]) for i in range(5)])
        self.final = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        for res in self.res_blocks:
            x = res(x)
        x = self.final(F.adaptive_avg_pool2d(x, 1))
        return x.squeeze()