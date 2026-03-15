from typing import List, Sequence, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from resblock import ResBlock
from style_encoder import StyleEncoder
from content_enc import ContentEncoder
from load_data import Q,P
from load_data import CHANNELS


class Generator(nn.Module):
    def __init__(self, in_channels=CHANNELS * 2):
        super().__init__()
        self.res1 = ResBlock(in_channels)
        self.res2 = ResBlock(in_channels)
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(in_channels, CHANNELS, 3, padding=1), nn.ReLU(inplace=True))  # 1024->512
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(CHANNELS, CHANNELS // 2, 3, padding=1), nn.ReLU(inplace=True))  # 512->256
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(CHANNELS // 2, CHANNELS // 4, 3, padding=1), nn.ReLU(inplace=True))  # 256->128
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(CHANNELS // 4, CHANNELS // 8, 3, padding=1), nn.ReLU(inplace=True))  # 128->64
        self.up5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(CHANNELS // 8, CHANNELS // 16, 3, padding=1), nn.ReLU(inplace=True))  # 64->32
        self.final = nn.Sequential(nn.Conv2d(CHANNELS // 16, 1, 3, padding=1), nn.Tanh())  # 32->1

    def forward(self, F, alphas_betas):
        x = self.res1(F, *alphas_betas[:4])
        x = self.res2(x, *alphas_betas[4:])
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.final(x)

class GANWritingGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = StyleEncoder()
        self.C = ContentEncoder()
        self.G = Generator()

    def forward(self, t, Xi):
        Fs = self.S(Xi)
        Fc, fc = self.C(t)
        hat_Fs = Fs + torch.randn_like(Fs)
        F = torch.cat((hat_Fs, Fc), dim=1)
        alphas_betas = [fc[:, i * Q : (i+1) * Q].unsqueeze(2).unsqueeze(3) for i in range(2 * P)]
        return self.G(F, alphas_betas)

