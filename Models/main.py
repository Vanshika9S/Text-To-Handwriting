import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
from writer_dataset import IAMDataset
from gen import GANWritingGenerator
from discriminator import Discriminator
from style_loss import StyleClassifier
from recognizer import Recognizer
from load_data import BATCH_SIZE,K,IMAGE_HEIGHT,IMAGE_WIDTH,MAX_L,ALPHABET_SIZE,CHAR_TO_IDX
from gen_word import generate_word
import kagglehub
raseshshettyiitbhu_word_iam_dataset_path = kagglehub.dataset_download('raseshshettyiitbhu/word-iam-dataset')
kavyarambhia_iam_words_csv_path = kagglehub.dataset_download('kavyarambhia/iam-words-csv')
csv_path = os.path.join(kavyarambhia_iam_words_csv_path, 'cleaned_csv_dataset (1).csv')
word_path = os.path.join(raseshshettyiitbhu_word_iam_dataset_path, 'phosc_data/words') 


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available. Please run on a GPU-enabled environment.")
    df = pd.read_csv(csv_path)
    old_prefix = "/kaggle/input/word-iam-dataset/phosc_data/words"
    new_prefix = "/root/.cache/kagglehub/datasets/raseshshettyiitbhu/word-iam-dataset/versions/1/phosc_data/words"
    df["Filepath"] = df["Filepath"].str.replace(old_prefix, new_prefix, regex=False)
    df['Text_Length'] = df['Text'].apply(lambda x: len(str(x).strip()))
    df = df[df['Text_Length'] >= 4].reset_index(drop=True)
    df
    if df.empty:
        raise RuntimeError("No data points remain after filtering for word length >= 4.") 
    print(f"Dataset initialized with {len(df)} samples (words of length >= 4).")
    dataset = IAMDataset(df, word_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)
    num_writers = len(dataset.writers)
    H = GANWritingGenerator().to(device)
    D = Discriminator().to(device)
    W = StyleClassifier(num_writers).to(device)
    R = Recognizer().to(device)
    adv_loss = nn.BCEWithLogitsLoss().to(device)
    style_loss = nn.CrossEntropyLoss().to(device)
    opt_H = torch.optim.Adam(H.parameters(), lr=1e-4)
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4)
    opt_W = torch.optim.Adam(W.parameters(), lr=1e-4)
    opt_R = torch.optim.Adam(R.parameters(), lr=1e-4)
    
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('generated_images', exist_ok=True)
    
    sample_word = "hello"
    
    for epoch in tqdm(range(20), desc='Epochs'):
        with tqdm(dataloader, total=len(dataloader), desc="Batches", leave=True, dynamic_ncols=True, position=0) as batch_pbar:
            for Xi, x, t, wi in batch_pbar:
                Xi, x, t, wi = Xi.to(device), x.to(device), t.to(device), wi.to(device)
                assert Xi.shape[1:] == (K, IMAGE_HEIGHT, IMAGE_WIDTH), f"Xi shape: {Xi.shape[1:]}"
                assert x.shape[1:] == (1, IMAGE_HEIGHT, IMAGE_WIDTH), f"x shape: {x.shape[1:]}"
                assert t.shape[1:] == (MAX_L, ALPHABET_SIZE), f"t shape: {t.shape[1:]}"
                bar_x = H(t, Xi)
                assert bar_x.shape[1:] == (1, IMAGE_HEIGHT, IMAGE_WIDTH), f"bar_x shape: {bar_x.shape[1:]}"
                real = D(x)
                fake = D(bar_x.detach())
                loss_D = adv_loss(real, torch.ones_like(real, device=device)) + adv_loss(fake, torch.zeros_like(fake, device=device))
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                pred_w = W(x)
                loss_W = style_loss(pred_w, wi)
                opt_W.zero_grad()
                loss_W.backward()
                opt_W.step()
                pred_r = R(x, t, 1.0)
                t_labels = t.argmax(-1).view(-1)
                loss_R = F.cross_entropy(pred_r.view(-1, ALPHABET_SIZE), t_labels, ignore_index=CHAR_TO_IDX['ε'])
                opt_R.zero_grad()
                loss_R.backward()
                opt_R.step()
                fake = D(bar_x)
                pred_w = W(bar_x)
                pred_r = R(bar_x, t, 1.0)
                loss_adv = adv_loss(fake, torch.ones_like(fake, device=device))
                loss_style = style_loss(pred_w, wi)
                loss_content = F.cross_entropy(pred_r.view(-1, ALPHABET_SIZE), t_labels, ignore_index=CHAR_TO_IDX['ε'])
                loss_H = loss_adv + loss_style + loss_content
                opt_H.zero_grad()
                loss_H.backward()
                opt_H.step()
                batch_pbar.set_postfix({'loss_H': f"{loss_H.item():.4f}", 'loss_D': f"{loss_D.item():.4f}", 'loss_W': f"{loss_W.item():.4f}", 'loss_R': f"{loss_R.item():.4f}"})
            
            with torch.no_grad():
                sample_Xi, _, sample_t, _ = next(iter(dataloader))
                sample_Xi, sample_t = sample_Xi.to(device), sample_t.to(device)
                sample_bar_x = H(sample_t, sample_Xi)
                for i in range(min(5, BATCH_SIZE)):
                    save_image(sample_bar_x[i], f'generated_images/epoch_{epoch+1}_sample_{i+1}.png', normalize=True)
            
            generate_word(sample_word, dataset, H, device, epoch=epoch)
            
            if (epoch + 1) % 5 == 0:
                torch.save(H.state_dict(), f'checkpoints/ganwriting_generator_epoch_{epoch+1}.pth')
                torch.save(D.state_dict(), f'checkpoints/discriminator_epoch_{epoch+1}.pth')
                torch.save(W.state_dict(), f'checkpoints/style_classifier_epoch_{epoch+1}.pth')
                torch.save(R.state_dict(), f'checkpoints/recognizer_epoch_{epoch+1}.pth')