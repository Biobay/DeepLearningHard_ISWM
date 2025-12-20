import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Imports
from config import *
from dataset import get_loader
from model import get_model
from loss_utils import PerceptualLoss # NEW

def add_noise(img, noise_factor=0.1):
    """Adds Gaussian noise to the image"""
    noise = torch.randn_like(img) * noise_factor
    noisy_img = img + noise
    return torch.clamp(noisy_img, 0., 1.)

def train():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Dataset & Loader
    print(f"Loading data from: {TRAIN_IMAGES_PATH}")
    loader = get_loader(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, BATCH_SIZE, is_train=True)

    # 2. Model (U-Net Denoising AE)
    model = get_model(device, latent_dim=LATENT_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. New Loss Function
    criterion = PerceptualLoss(device)

    print(f"Starting Denoising Training for {NUM_EPOCHS} epochs...")
    
    best_loss = float('inf')
    loss_history = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for img, _ in pbar:
            img = img.to(device)
            
            # --- DENOISING LOGIC ---
            # Input = Noisy Image
            noisy_input = add_noise(img, noise_factor=0.2)
            
            # Target = Clean Image
            target = img

            # Forward
            optimizer.zero_grad()
            recon = model(noisy_input) # Try to clean the noise
            
            # Loss (MAE + VGG)
            loss = criterion(recon, target)
            
            # Backward
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        # Save Best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            
        # Visualization (Sanity Check)
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample = noisy_input[0].cpu().permute(1,2,0)
                clean = recon[0].cpu().permute(1,2,0)
                plt.figure(figsize=(8,4))
                plt.subplot(1,2,1); plt.imshow(sample); plt.title("Noisy Input")
                plt.subplot(1,2,2); plt.imshow(clean); plt.title("Restored Output")
                plt.savefig(f"epoch_{epoch+1}_check.png")
                plt.close()

    print(f"Training Complete! Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
