import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
import random
from tqdm import tqdm
import os

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;'*!?()[]/-\" ε"
ALPHABET_SIZE = len(ALPHABET)
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
MAX_L = 10
K = 15
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 256
CHANNELS = 512
P = 4
Q = CHANNELS * 2  
N = 256
M = 512
R = 2
BATCH_SIZE = 64 