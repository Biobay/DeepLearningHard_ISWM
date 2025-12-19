
import torch
import os
import csv
import numpy as np
from PIL import Image
from dataset import get_loader
from model import get_model
from config import *
import matplotlib.pyplot as plt

def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    
    model = get_model(device=device, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE[0])
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        print("Model not found, skipping inference.")
        return

    model.eval()
    loader = get_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, 1, is_train=False)
    results = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, (img, mask, name) in enumerate(loader):
            img = img.to(device)
            recon = model(img)
            
            # Anomaly Map
            diff = torch.abs(img - recon).mean(dim=1)
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            binary = (diff > ANOMALY_THRESHOLD).float().cpu().numpy().squeeze()
            
            # Classification
            has_crack = 1 if np.sum(binary) > 100 else 0
            results.append((name[0], has_crack))
            
            # Save Mask
            Image.fromarray((binary*255).astype(np.uint8)).save(os.path.join(PREDICTIONS_PATH, name[0]))

            # Save Visualization for the first image
            if i == 0:
                plt.figure(figsize=(10,4))
                plt.subplot(1,4,1); plt.imshow(img.cpu().squeeze().permute(1,2,0)); plt.title("Input")
                plt.subplot(1,4,2); plt.imshow(recon.cpu().squeeze().permute(1,2,0)); plt.title("Recon")
                plt.subplot(1,4,3); plt.imshow(diff.cpu().squeeze(), cmap='jet'); plt.title("Diff")
                plt.subplot(1,4,4); plt.imshow(binary, cmap='gray'); plt.title("Pred")
                plt.savefig("results_visualization.png")
                
    with open(os.path.join(PREDICTIONS_PATH, 'classification_results.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'has_crack'])
        writer.writerows(results)
    print("Inference Done.")

if __name__ == "__main__":
    inference()
