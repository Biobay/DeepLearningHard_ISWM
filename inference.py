"""
Script di inference per Anomaly Detection.

Processo:
1. Carica modello addestrato
2. Passa immagini di test attraverso l'autoencoder
3. Calcola differenza tra originale e ricostruita
4. Applica threshold per generare maschera binaria
5. Salva predizioni per submission Kaggle
"""

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

from model import get_model
from dataset import get_test_loader, get_data_transforms
from config import *

def compute_anomaly_map(original, reconstructed):
    """
    Calcola la mappa di anomalia dalla differenza tra immagine originale e ricostruita.
    
    Args:
        original: immagine originale [B, 3, H, W]
        reconstructed: immagine ricostruita [B, 3, H, W]
    
    Returns:
        anomaly_map: mappa di differenza [B, 1, H, W] in range [0, 1]
    """
    # Differenza assoluta pixel-wise
    diff = torch.abs(original - reconstructed)
    
    # Media sui canali RGB -> grayscale
    anomaly_map = torch.mean(diff, dim=1, keepdim=True)
    
    # Normalizza in [0, 1]
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    return anomaly_map

def apply_threshold(anomaly_map, threshold=0.1):
    """
    Binarizza la mappa di anomalia usando un threshold.
    
    Args:
        anomaly_map: mappa continua [B, 1, H, W]
        threshold: soglia di binarizzazione
    
    Returns:
        binary_mask: maschera binaria [B, 1, H, W] con valori {0, 1}
    """
    binary_mask = (anomaly_map > threshold).float()
    return binary_mask

def post_process_mask(mask, min_area=50):
    """
    Post-processing della maschera per rimuovere rumore.
    
    Args:
        mask: numpy array [H, W] binario
        min_area: area minima per considerare una componente connessa
    
    Returns:
        cleaned_mask: maschera pulita
    """
    # Converti in uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Morfologia: chiusura (riempie piccoli buchi)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Rimuovi piccole componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_closed, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask / 255.0  # Ritorna in [0, 1]

def inference():
    """
    Esegue inference sul test set e genera le maschere predette.
    """
    # Setup device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Crea cartella per salvare predizioni
    os.makedirs(PREDICTIONS_PATH, exist_ok=True)
    
    # Carica modello addestrato
    print(f"\nCaricamento modello da {BEST_MODEL_PATH}...")
    model = get_model(device=device, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()  # Modalità evaluation
    print("Modello caricato con successo!")
    
    # Carica test set
    print(f"\nCaricamento test set da {TEST_IMAGES_PATH}...")
    test_loader = get_test_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, batch_size=1)  # Batch=1 per salvare singolarmente
    print(f"Test set caricato: {len(test_loader.dataset)} immagini")
    
    print(f"\n{'='*50}")
    print(f"Inizio Inference - Generazione maschere predette")
    print(f"{'='*50}\n")
    
    # Lista per salvare risultati
    all_predictions = []
    all_ground_truths = []
    
    with torch.no_grad():  # Disabilita gradients
        for images, masks, filenames in tqdm(test_loader, desc="Inference"):
            # Sposta su device
            images = images.to(device)
            
            # Forward pass: ricostruzione
            reconstructed = model(images)
            
            # Calcola mappa di anomalia
            anomaly_map = compute_anomaly_map(images, reconstructed)
            
            # Applica threshold per binarizzare
            binary_mask = apply_threshold(anomaly_map, threshold=ANOMALY_THRESHOLD)
            
            # Post-processing (opzionale)
            mask_np = binary_mask.squeeze().cpu().numpy()
            mask_cleaned = post_process_mask(mask_np, min_area=50)
            
            # Salva maschera predetta
            filename = filenames[0]
            save_path = os.path.join(PREDICTIONS_PATH, filename)
            
            # Converti in PIL Image e salva
            mask_img = Image.fromarray((mask_cleaned * 255).astype(np.uint8))
            mask_img.save(save_path)
            
            # Accumula per evaluation
            all_predictions.append(mask_cleaned)
            all_ground_truths.append(masks.squeeze().cpu().numpy())
    
    print(f"\n{'='*50}")
    print(f"Inference completata!")
    print(f"Maschere salvate in: {PREDICTIONS_PATH}")
    print(f"{'='*50}\n")
    
    return all_predictions, all_ground_truths

def visualize_results(num_samples=5):
    """
    Visualizza alcuni esempi di predizioni.
    """
    import matplotlib.pyplot as plt
    
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Carica modello
    model = get_model(device=device, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    # Carica test set
    test_loader = get_test_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, batch_size=1)
    
    # Seleziona alcuni campioni random
    samples = []
    for i, (images, masks, filenames) in enumerate(test_loader):
        if i >= num_samples:
            break
        samples.append((images, masks, filenames))
    
    # Crea figure
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    
    with torch.no_grad():
        for idx, (images, masks, filenames) in enumerate(samples):
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            
            # Anomaly map
            anomaly_map = compute_anomaly_map(images, reconstructed)
            binary_mask = apply_threshold(anomaly_map, threshold=ANOMALY_THRESHOLD)
            
            # Denormalizza per visualizzazione
            img_denorm = (images.squeeze().cpu().permute(1, 2, 0).numpy() + 1) / 2
            recon_denorm = reconstructed.squeeze().cpu().permute(1, 2, 0).numpy()
            
            # Plot
            axes[idx, 0].imshow(img_denorm)
            axes[idx, 0].set_title('Originale')
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(recon_denorm)
            axes[idx, 1].set_title('Ricostruita')
            axes[idx, 1].axis('off')
            
            axes[idx, 2].imshow(anomaly_map.squeeze().cpu(), cmap='hot')
            axes[idx, 2].set_title('Anomaly Map')
            axes[idx, 2].axis('off')
            
            axes[idx, 3].imshow(binary_mask.squeeze().cpu(), cmap='gray')
            axes[idx, 3].set_title('Predizione')
            axes[idx, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=150, bbox_inches='tight')
    print("Visualizzazione salvata in: results_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Esegui inference
    predictions, ground_truths = inference()
    
    # Visualizza alcuni risultati
    print("\nGenerazione visualizzazione...")
    visualize_results(num_samples=5)
