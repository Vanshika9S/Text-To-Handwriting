import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
import random
from load_data import ALPHABET,ALPHABET_SIZE, IMAGE_HEIGHT, MAX_L

class Recognizer(nn.Module):
    def __init__(self, alphabet_size=ALPHABET_SIZE):
        super().__init__()
        vgg = models.vgg19_bn(pretrained=True)
        vgg.features[0] = nn.Conv2d(1, 64, 3, padding=1)
        self.encoder = vgg.features
        self.h_prime = IMAGE_HEIGHT // 32
        self.input_size = 512 * self.h_prime
        self.bgru = nn.GRU(self.input_size, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.decoder = nn.GRU(alphabet_size + 512, 512, batch_first=True)
        self.attention = nn.Sequential(nn.Linear(512 + 512, 512), nn.Tanh(), nn.Linear(512, 1))
        self.out = nn.Linear(512, alphabet_size)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        feat = self.encoder(x)
        b, c, h, w = feat.size()
        assert h == self.h_prime, f"Expected h={self.h_prime}, got {h}"
        feat = feat.view(b, c * h, w).permute(0, 2, 1)
        assert feat.size(-1) == self.input_size, f"Expected feat last dim {self.input_size}, got {feat.size(-1)}"
        encoder_outputs, hidden = self.bgru(feat)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1).unsqueeze(0)
        outputs = []
        input = torch.zeros(b, 1, ALPHABET_SIZE).to(x.device)
        for t_step in range(MAX_L):
            hidden_exp = hidden[0].unsqueeze(1).repeat(1, w, 1)
            att = self.attention(torch.cat((hidden_exp, encoder_outputs), dim=2)).squeeze(2)
            att_weights = F.softmax(att, dim=1)
            context = (att_weights.unsqueeze(2) * encoder_outputs).sum(dim=1)
            dec_in = torch.cat((input.squeeze(1), context), dim=1).unsqueeze(1)
            output, hidden = self.decoder(dec_in, hidden)
            pred = self.out(output.squeeze(1))
            outputs.append(pred)
            if random.random() < teacher_forcing_ratio and target is not None:
                input = target[:, t_step, :].unsqueeze(1)
            else:
                input = F.softmax(pred, dim=1).detach().unsqueeze(1)
        return torch.stack(outputs, dim=1)