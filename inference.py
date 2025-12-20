import torch
import os
import csv
import cv2
import numpy as np
from PIL import Image
from dataset import get_loader
from model import get_model
from config import *
import matplotlib.pyplot as plt

def clean_mask_pro(mask_np, min_blob_area):
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    output_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_blob_area:
            output_mask[labels == i] = 255
    return (output_mask > 127).astype(float)

def inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    
    model = get_model(device=device, latent_dim=LATENT_DIM)
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
        print("Loaded Simple CAE Model.")
    else:
        print("Model not found. Please train first.")
        return

    model.eval()
    loader = get_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, 1, is_train=False)
    results = []
    
    # --- THE SWEET SPOT ---
    FINAL_THRESHOLD = 0.25
    FINAL_MIN_AREA = 90
    
    print(f"Inference: Threshold={FINAL_THRESHOLD}, MinArea={FINAL_MIN_AREA}")
    
    with torch.no_grad():
        for i, (img, mask, name) in enumerate(loader):
            img = img.to(device)
            recon = model(img)
            
            diff = torch.abs(img - recon).mean(dim=1)
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            diff_np = diff.cpu().numpy().squeeze()
            
            binary = (diff_np > FINAL_THRESHOLD).astype(float)
            clean = clean_mask_pro(binary, min_blob_area=FINAL_MIN_AREA)
            
            has_crack = 1 if np.sum(clean) > 0 else 0
            results.append((name[0], has_crack))
            
            save_path = os.path.join(PREDICTIONS_PATH, name[0])
            Image.fromarray((clean*255).astype(np.uint8)).save(save_path)

            if i < 5:
                plt.figure(figsize=(12, 4))
                plt.subplot(1,4,1); plt.imshow(img.cpu().squeeze().permute(1,2,0)); plt.title("Input")
                plt.subplot(1,4,2); plt.imshow(recon.cpu().squeeze().permute(1,2,0)); plt.title("Recon (Blurry=Good)")
                plt.subplot(1,4,3); plt.imshow(diff_np, cmap='jet'); plt.title("Diff Map")
                plt.subplot(1,4,4); plt.imshow(clean, cmap='gray'); plt.title(f"Pred (Crack: {has_crack})")
                plt.savefig(f"final_hero_{i}.png")
                plt.close()

    print("Inference Complete.")

if __name__ == "__main__":
    inference()
