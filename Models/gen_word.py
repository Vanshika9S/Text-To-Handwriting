import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.utils import save_image
from PIL import Image
import random
import os
from load_data import ALPHABET,ALPHABET_SIZE,MAX_L,N,M,K,P,Q,R,IMAGE_HEIGHT,IMAGE_WIDTH,CHAR_TO_IDX

def generate_word(word, dataset, model, device, output_dir='generated_images', epoch=0):
    model.eval()
    with torch.no_grad():
        t = torch.zeros(1, MAX_L, ALPHABET_SIZE).to(device)
        len_word = min(len(word), MAX_L)
        for i in range(len_word):
            if word[i] in CHAR_TO_IDX:
                t[0, i, CHAR_TO_IDX[word[i]]] = 1
        t[0, len_word:, CHAR_TO_IDX['ε']] = 1
        
        writer = random.choice(dataset.writers)
        group = dataset.writer_groups.get_group(writer)
        samples = group.sample(K, replace=len(group) < K)
        Xi_paths = samples['Filepath'].tolist()
        Xi = torch.stack([dataset.transform(Image.open(path)) for path in Xi_paths]).squeeze(1)  
        Xi = Xi.unsqueeze(0).to(device)  
        
        bar_x = model(t, Xi)  
        save_image(bar_x[0], os.path.join(output_dir, f'epoch_{epoch+1}_word_{word}.png'), normalize=True)
    model.train()
 