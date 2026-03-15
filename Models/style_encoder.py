from torchvision import models
import torch 
import torch.nn as nn
import torch.nn.functional as F
from load_data import K

class StyleEncoder(nn.Module):
    def __init__(self, k=K):
        super().__init__()
        vgg = models.vgg19_bn(pretrained=True)
        vgg.features[0] = nn.Conv2d(k, 64, 3, padding=1)
        self.features = vgg.features

    def forward(self, Xi):
        return self.features(Xi)