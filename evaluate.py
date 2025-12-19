"""
Script di evaluation per calcolare metriche di segmentazione.

Metriche implementate:
- IoU (Intersection over Union / Jaccard Index)
- Dice Coefficient (F1-Score)
- Precision
- Recall
- Pixel Accuracy
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_loader
from config import *

def calculate_iou(pred, target, smooth=1e-8):
    """
    Calcola IoU (Intersection over Union).
    
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred: predizione binaria [H, W]
        target: ground truth binaria [H, W]
        smooth: smoothing per evitare divisione per zero
    
    Returns:
        iou: valore IoU in [0, 1]
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou

def calculate_dice(pred, target, smooth=1e-8):
    """
    Calcola Dice Coefficient (equivalente a F1-Score).
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred: predizione binaria [H, W]
        target: ground truth binaria [H, W]
        smooth: smoothing per evitare divisione per zero
    
    Returns:
        dice: valore Dice in [0, 1]
    """
    intersection = np.logical_and(pred, target).sum()
    
    dice = (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice

def calculate_precision_recall(pred, target, smooth=1e-8):
    """
    Calcola Precision e Recall.
    
    Precision = TP / (TP + FP)  -> Quanto sono accurate le predizioni positive
    Recall = TP / (TP + FN)     -> Quanto riesco a trovare dei veri positivi
    
    Args:
        pred: predizione binaria [H, W]
        target: ground truth binaria [H, W]
        smooth: smoothing
    
    Returns:
        precision, recall
    """
    true_positive = np.logical_and(pred == 1, target == 1).sum()
    false_positive = np.logical_and(pred == 1, target == 0).sum()
    false_negative = np.logical_and(pred == 0, target == 1).sum()
    
    precision = (true_positive + smooth) / (true_positive + false_positive + smooth)
    recall = (true_positive + smooth) / (true_positive + false_negative + smooth)
    
    return precision, recall

def calculate_pixel_accuracy(pred, target):
    """
    Calcola Pixel Accuracy.
    
    Accuracy = Pixel corretti / Pixel totali
    """
    correct = (pred == target).sum()
    total = pred.size
    
    accuracy = correct / total
    
    return accuracy

def evaluate_model(threshold=ANOMALY_THRESHOLD):
    """
    Valuta il modello sul test set usando le maschere ground truth.
    
    Args:
        threshold: soglia per binarizzare anomaly map
    
    Returns:
        metrics: dizionario con tutte le metriche
    """
    # Setup device
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Carica modello
    print(f"\nCaricamento modello da {BEST_MODEL_PATH}...")
    img_size = IMAGE_SIZE[0]
    model = get_model(device=device, latent_dim=LATENT_DIM, image_size=img_size)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    # Carica test set
    print(f"\nCaricamento test set...")
    test_loader = get_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, batch_size=8, is_train=False)
    
    print(f"\n{'='*50}")
    print(f"Evaluation - Threshold: {threshold}")
    print(f"{'='*50}\n")
    
    # Accumulatori per metriche
    all_ious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []
    
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc="Evaluating"):
            # Sposta su device
            images = images.to(device)
            
            # Forward pass
            reconstructed = model(images)
            
            # Anomaly detection
            # Calcul de la carte d'anomalie
            diff = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            binary_pred = (diff > threshold).float()
            
            # Converti in numpy
            pred_np = binary_pred.cpu().numpy()
            target_np = masks.cpu().numpy()
            
            # Calcola metriche per ogni immagine del batch
            for i in range(pred_np.shape[0]):
                pred_i = pred_np[i, 0]  # [H, W]
                target_i = target_np[i, 0]  # [H, W]
                
                iou = calculate_iou(pred_i, target_i)
                dice = calculate_dice(pred_i, target_i)
                precision, recall = calculate_precision_recall(pred_i, target_i)
                accuracy = calculate_pixel_accuracy(pred_i, target_i)
                
                all_ious.append(iou)
                all_dices.append(dice)
                all_precisions.append(precision)
                all_recalls.append(recall)
                all_accuracies.append(accuracy)
    
    # Calcola statistiche
    metrics = {
        'IoU': {
            'mean': np.mean(all_ious),
            'std': np.std(all_ious),
            'median': np.median(all_ious)
        },
        'Dice': {
            'mean': np.mean(all_dices),
            'std': np.std(all_dices),
            'median': np.median(all_dices)
        },
        'Precision': {
            'mean': np.mean(all_precisions),
            'std': np.std(all_precisions),
            'median': np.median(all_precisions)
        },
        'Recall': {
            'mean': np.mean(all_recalls),
            'std': np.std(all_recalls),
            'median': np.median(all_recalls)
        },
        'Accuracy': {
            'mean': np.mean(all_accuracies),
            'std': np.std(all_accuracies),
            'median': np.median(all_accuracies)
        }
    }
    
    # Print risultati
    print(f"\n{'='*50}")
    print(f"RISULTATI EVALUATION")
    print(f"{'='*50}\n")
    
    for metric_name, stats in metrics.items():
        print(f"{metric_name}:")
        print(f"  Mean:   {stats['mean']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Std:    {stats['std']:.4f}")
        print()
    
    return metrics, all_ious, all_dices

def find_best_threshold():
    """
    Trova il miglior threshold testando diversi valori.
    Usa IoU come metrica di riferimento.
    """
    # Setup
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    img_size = IMAGE_SIZE[0]
    model = get_model(device=device, latent_dim=LATENT_DIM, image_size=img_size)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.eval()
    
    test_loader = get_loader(TEST_IMAGES_PATH, TEST_MASKS_PATH, batch_size=8, is_train=False)
    
    # Test diversi threshold
    thresholds = np.linspace(0.01, 0.5, 20)
    mean_ious = []
    
    print(f"\n{'='*50}")
    print(f"Ricerca Best Threshold")
    print(f"{'='*50}\n")
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        ious = []
        
        with torch.no_grad():
            for images, masks, _ in test_loader:
                images = images.to(device)
                
                reconstructed = model(images)
                # Calcul de la carte d'anomalie
                diff = torch.abs(images - reconstructed).mean(dim=1, keepdim=True)
                diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                binary_pred = (diff > threshold).float()
                
                pred_np = binary_pred.cpu().numpy()
                target_np = masks.cpu().numpy()
                
                for i in range(pred_np.shape[0]):
                    iou = calculate_iou(pred_np[i, 0], target_np[i, 0])
                    ious.append(iou)
        
        mean_ious.append(np.mean(ious))
    
    # Trova threshold ottimale
    best_idx = np.argmax(mean_ious)
    best_threshold = thresholds[best_idx]
    best_iou = mean_ious[best_idx]
    
    print(f"\nBest Threshold: {best_threshold:.4f}")
    print(f"IoU@best_threshold: {best_iou:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mean_ious, 'b-', linewidth=2)
    plt.scatter([best_threshold], [best_iou], color='red', s=100, zorder=5, label=f'Best: {best_threshold:.4f}')
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Mean IoU', fontsize=12)
    plt.title('IoU vs Threshold', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("Plot salvato in: threshold_optimization.png")
    plt.show()
    
    return best_threshold, best_iou

if __name__ == "__main__":
    # 1. Valuta con threshold default
    print("=== EVALUATION CON THRESHOLD DEFAULT ===")
    metrics, ious, dices = evaluate_model(threshold=ANOMALY_THRESHOLD)
    
    # 2. Trova threshold ottimale
    print("\n\n=== OTTIMIZZAZIONE THRESHOLD ===")
    best_threshold, best_iou = find_best_threshold()
    
    # 3. Valuta con threshold ottimale
    print(f"\n\n=== EVALUATION CON BEST THRESHOLD ({best_threshold:.4f}) ===")
    metrics_optimized, _, _ = evaluate_model(threshold=best_threshold)
