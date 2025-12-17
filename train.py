"""
Script di training per Convolutional Autoencoder.

Strategia:
1. Carica SOLO le immagini (no maschere)
2. Addestra l'autoencoder a ricostruire le immagini
3. Loss = MSE tra input e output (Mean Squared Error)
4. Salva il modello migliore
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import numpy as np

from model import get_model
from dataset import get_train_loader
from config import *

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
    
    # Loss function: MSE (Mean Squared Error)
    # Penalizza la differenza quadratica media tra pixel
    criterion = nn.MSELoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler: riduce LR quando la loss non migliora
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # Dimezza LR
        patience=5,  # Aspetta 5 epoche senza miglioramenti
        verbose=True
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
