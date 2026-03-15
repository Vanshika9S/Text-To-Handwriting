import torch
import torch.nn as nn
from torchvision import models
from discriminator import DiscriminatorResBlock
import torch.nn.functional as F

class StyleClassifier(nn.Module):
    def __init__(self, num_writers):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        channels = [64, 128, 256, 512, 512, 512]
        self.res_blocks = nn.ModuleList([DiscriminatorResBlock(channels[i], channels[i+1]) for i in range(5)])
        self.final = nn.Linear(512, num_writers)

    def forward(self, x):
        x = self.conv1(x)
        for res in self.res_blocks:
            x = res(x)
        return self.final(F.adaptive_avg_pool2d(x, 1).flatten(1))