"""
Script di training per Convolutional Autoencoder.

Strategia:
1. Carica SOLO le immagini (no maschere)
2. Addestra l'autoencoder a ricostruire le immagini
3. Loss = MSE + SSIM (Structural Similarity) per migliore percezione
4. Salva il modello migliore
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from model import get_model
from dataset import get_train_loader
from config import *


class SSIMLoss(nn.Module):
    """SSIM Loss per migliore percezione delle strutture."""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self._create_window(window_size, self.channel)
        
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()
    
    def _create_window(self, window_size, channel):
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, self.window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, self.window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self.window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self.window, padding=self.window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if self.size_average:
            return 1 - ssim_map.mean()  # 1 - SSIM per minimizzare
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class CombinedLoss(nn.Module):
    """Combinazione di MSE e SSIM Loss."""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.alpha = alpha  # peso tra MSE e SSIM
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss

def denormalize(tensor):
    """
    Inverte la normalizzazione per visualizzare le immagini.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    Training per una singola epoca.
    
    Returns:
        avg_loss: loss media dell'epoca
    """
    model.train()  # Modalità training
    epoch_loss = 0.0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # Sposta dati su GPU/CPU
        images = images.to(device)
        targets = targets.to(device)  # targets = images (autoencoder!)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: ricostruzione
        reconstructed = model(images)
        
        # Calcola loss: quanto è diversa la ricostruzione dall'originale?
        # MSE (Mean Squared Error) penalizza differenze pixel-wise
        loss = criterion(reconstructed, targets)
        
        # Backward pass: calcola gradienti
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Accumula loss
        epoch_loss += loss.item()
        
        # Aggiorna progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Loss media dell'epoca
    avg_loss = epoch_loss / len(train_loader)
    
    return avg_loss

def validate(model, val_loader, criterion, device):
    """
    Validazione del modello (opzionale, se hai un validation set).
    Per ora usiamo tutto il train set.
    """
    model.eval()  # Modalità evaluation
    val_loss = 0.0
    
    with torch.no_grad():  # Disabilita gradients (velocizza e risparmia memoria)
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            reconstructed = model(images)
            
            # Calcola loss
            loss = criterion(reconstructed, targets)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    
    return avg_val_loss

def train():
    """
    Loop di training principale.
    """
    # Setup device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crea cartelle per salvare modelli
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Carica dataset (SOLO immagini, no maschere!)
    print(f"\nCaricamento dataset da {TRAIN_IMAGES_PATH}...")
    train_loader = get_train_loader(TRAIN_IMAGES_PATH, batch_size=BATCH_SIZE)
    print(f"Dataset caricato: {len(train_loader.dataset)} immagini")
    
    # Crea modello
    print("\nCreazione modello...")
    model = get_model(device=device, latent_dim=LATENT_DIM)
    
    # Loss function: MSE + SSIM combinata
    # MSE per differenze pixel-wise, SSIM per strutture percettive
    criterion = CombinedLoss(alpha=0.5)  # 50% MSE, 50% SSIM
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: riduce LR quando la loss non migliora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # Dimezza LR
        patience=5  # Aspetta 5 epoche senza miglioramenti
    )
    
    # Tensorboard per visualizzare training
    writer = SummaryWriter('runs/autoencoder_training')
    
    # Tracking best model
    best_loss = float('inf')
    
    print(f"\n{'='*50}")
    print(f"Inizio Training - {NUM_EPOCHS} epoche")
    print(f"{'='*50}\n")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        # Training per un'epoca
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Log su tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}")
        
        # Update learning rate scheduler
        scheduler.step(train_loss)
        
        # Salva checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, checkpoint_path)
            print(f"Checkpoint salvato: {checkpoint_path}")
        
        # Salva best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"✓ Nuovo best model salvato! Loss: {best_loss:.6f}")
        
        print()
    
    print(f"\n{'='*50}")
    print(f"Training completato!")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Modello salvato in: {BEST_MODEL_PATH}")
    print(f"{'='*50}\n")
    
    writer.close()

if __name__ == "__main__":
    train()
