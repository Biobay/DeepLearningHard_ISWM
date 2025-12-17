"""
Script di training ottimizzato per cloud (Salad).
Salva automaticamente checkpoints e supporta interruzioni.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse
import signal
import sys

from model import get_model
from dataset import get_train_loader
from config import *

# Flag per gestire interruzioni
interrupted = False

def signal_handler(sig, frame):
    """Gestisce interruzioni per salvare modello prima di terminare."""
    global interrupted
    print('\n\nInterruzione rilevata! Salvataggio in corso...')
    interrupted = True

# Registra handler per SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

def save_checkpoint(model, optimizer, epoch, loss, filename):
    """Salva checkpoint completo."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"✓ Checkpoint salvato: {filename}")

def load_checkpoint(model, optimizer, filename):
    """Carica checkpoint per riprendere training."""
    if os.path.exists(filename):
        print(f"Caricamento checkpoint da {filename}...")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"✓ Checkpoint caricato: epoch {epoch+1}, loss {loss:.6f}")
        return epoch + 1
    return 0

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Training per una singola epoca."""
    global interrupted
    
    model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        if interrupted:
            print("\nInterruzione durante training...")
            break
        
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(images)
        loss = criterion(reconstructed, targets)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / len(train_loader)
    return avg_loss

def train(resume=False):
    """Loop di training principale con supporto per resume."""
    global interrupted
    
    # Setup device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Crea directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Carica dataset
    print(f"\nCaricamento dataset da {TRAIN_IMAGES_PATH}...")
    train_loader = get_train_loader(TRAIN_IMAGES_PATH, batch_size=BATCH_SIZE)
    print(f"Dataset caricato: {len(train_loader.dataset)} immagini")
    
    # Crea modello
    print("\nCreazione modello...")
    model = get_model(device=device, latent_dim=LATENT_DIM)
    
    # Loss e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Resume da checkpoint se richiesto
    start_epoch = 0
    resume_path = 'checkpoints/latest.pth'
    if resume and os.path.exists(resume_path):
        start_epoch = load_checkpoint(model, optimizer, resume_path)
    
    # Tensorboard
    writer = SummaryWriter('runs/autoencoder_training')
    
    # Tracking best model
    best_loss = float('inf')
    
    print(f"\n{'='*50}")
    print(f"Inizio Training - Epoche {start_epoch+1} to {NUM_EPOCHS}")
    print(f"{'='*50}\n")
    
    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        if interrupted:
            print("\nInterruzione rilevata, salvataggio finale...")
            save_checkpoint(model, optimizer, epoch, epoch_loss, resume_path)
            break
        
        # Training epoca
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Log tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f}")
        
        # Update scheduler
        scheduler.step(train_loss)
        
        # Salva checkpoint ogni 5 epoche + ultimo checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Salva sempre l'ultimo checkpoint (per resume)
        save_checkpoint(model, optimizer, epoch, train_loss, resume_path)
        
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

def main():
    parser = argparse.ArgumentParser(description='Training su Salad Cloud')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training da ultimo checkpoint')
    
    args = parser.parse_args()
    
    train(resume=args.resume)

if __name__ == "__main__":
    main()
