from typing import List, Sequence, Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import ALPHABET,ALPHABET_SIZE,MAX_L,N,M,P,Q,R,IMAGE_HEIGHT,IMAGE_WIDTH

class ContentEncoder(nn.Module):
    def __init__(self, alphabet_size=ALPHABET_SIZE, max_l=MAX_L, n=N, m=M, p=P, q=Q, r=R):
        super().__init__()
        self.emb = nn.Linear(alphabet_size, n)
        self.g1 = nn.Sequential(nn.Linear(n, m), nn.ReLU(), nn.BatchNorm1d(m), nn.Linear(m, m), nn.ReLU(), nn.BatchNorm1d(m), nn.Linear(m, m), nn.ReLU(), nn.BatchNorm1d(m))
        self.g2 = nn.Sequential(
            nn.Linear(max_l * n, max_l * n // 2), nn.ReLU(), nn.BatchNorm1d(max_l * n // 2),
            nn.Linear(max_l * n // 2, max_l * n // 4), nn.ReLU(), nn.BatchNorm1d(max_l * n // 4),
            nn.Linear(max_l * n // 4, 2 * p * q), nn.ReLU(), nn.BatchNorm1d(2 * p * q)
        )
        self.max_l = max_l
        self.m = m
        self.r = r
        self.h_prime = IMAGE_HEIGHT // 32
        self.w_prime = IMAGE_WIDTH // 32

    def forward(self, t):
        e = self.emb(t)
        g1_out = self.g1(e.view(-1, N)).view(t.size(0), self.max_l, self.m)
        Fc = torch.cat([g1_out[:, i, :].unsqueeze(2).unsqueeze(3).repeat(1, 1, self.h_prime, self.r) for i in range(self.max_l)], dim=3)
        Fc = F.interpolate(Fc, size=(self.h_prime, self.w_prime), mode='nearest')
        fc = self.g2(e.view(t.size(0), -1))
        return Fc, fc