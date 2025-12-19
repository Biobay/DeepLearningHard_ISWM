
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from model import get_model
from dataset import get_loader
from config import *

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, pred, target):
        return self.mse(pred, target)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading data from: {TRAIN_IMAGES_PATH}")
    loader = get_loader(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH, BATCH_SIZE, is_train=True)
    
    if len(loader.dataset) == 0:
        print("ERROR: Dataset is empty.")
        return

    model = get_model(device=device, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE[0])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CombinedLoss()
    
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for img, _ in pbar:
            img = img.to(device)
            optimizer.zero_grad()
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        # Save model every epoch to be safe
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        
    print(f"Training Complete! Model saved to {BEST_MODEL_PATH}")

if __name__ == "__main__":
    train()
