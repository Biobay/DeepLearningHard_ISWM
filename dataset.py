"""
Dataset personalizzato per caricare solo immagini (senza maschere durante training).
Approccio unsupervised per anomaly detection.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import glob
from config import IMAGE_SIZE

class CrackImageDataset(Dataset):
    """
    Dataset che carica SOLO le immagini (no maschere).
    L'autoencoder imparerà a ricostruire le immagini così come sono,
    fallendo sulle anomalie (crepe).
    """
    
    def __init__(self, images_dir, transform=None):
        """
        Args:
            images_dir: Path alla cartella con le immagini (es. train/images/)
            transform: Trasformazioni da applicare alle immagini
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # Carica tutti i path delle immagini (solo .jpg)
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        
        print(f"Trovate {len(self.image_paths)} immagini in {images_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Ritorna: (immagine, immagine)
        Perché? L'autoencoder deve ricostruire l'input stesso!
        Input = Target per calcolare la loss MSE
        """
        # Carica immagine
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Applica trasformazioni
        if self.transform:
            image = self.transform(image)
        
        # Ritorna (input, target) dove input=target
        # Questo è fondamentale per l'autoencoder!
        return image, image


class CrackTestDataset(Dataset):
    """
    Dataset per il test che include anche le maschere per valutazione.
    Le maschere NON vengono usate durante training, solo per calcolare metriche.
    """
    
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        Args:
            images_dir: Path alle immagini di test
            masks_dir: Path alle maschere di test (per evaluation)
            transform: Trasformazioni per immagini
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, '*.jpg')))
        
        # Le maschere hanno lo stesso nome delle immagini
        self.mask_paths = [
            os.path.join(masks_dir, os.path.basename(img_path))
            for img_path in self.image_paths
        ]
        
        print(f"Trovate {len(self.image_paths)} immagini di test")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Ritorna: (immagine, maschera, filename)
        """
        # Carica immagine
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Carica maschera (ground truth per evaluation)
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Trasformazioni
        if self.transform:
            image = self.transform(image)
            # Maschera: solo resize e to tensor (usa IMAGE_SIZE da config)
            mask = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor()
            ])(mask)
        
        # Binarizza maschera (0 o 1)
        mask = (mask > 0.5).float()
        
        filename = os.path.basename(img_path)
        
        return image, mask, filename


def get_data_transforms():
    """
    Definisce le trasformazioni per preprocessing immagini.
    Normalizzazione standard per reti neurali.
    Usa IMAGE_SIZE da config.py (128x128 o 256x256).
    """
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # Resize da config
        transforms.ToTensor(),  # Converte in tensor [0,1]
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalizzazione ImageNet standard
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform


def get_train_loader(train_images_dir, batch_size=32):
    """
    Crea DataLoader per training (SOLO immagini, no maschere).
    """
    transform = get_data_transforms()
    train_dataset = CrackImageDataset(train_images_dir, transform=transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle per training
        num_workers=2,  # Ridotto per evitare problemi shared memory su cloud
        pin_memory=True  # Velocizza transfer GPU
    )
    
    return train_loader


def get_test_loader(test_images_dir, test_masks_dir, batch_size=32):
    """
    Crea DataLoader per test (immagini + maschere per evaluation).
    """
    transform = get_data_transforms()
    test_dataset = CrackTestDataset(test_images_dir, test_masks_dir, transform=transform)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle per test
        num_workers=2,  # Ridotto per cloud
        pin_memory=True
    )
    
    return test_loader
