
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from config import IMAGE_SIZE

class CrackTrainDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform  # <--- THIS LINE WAS MISSING, NOW FIXED
        self.image_files = []
        
        # PRO FILTERING: Only healthy images
        if os.path.exists(images_dir):
            all_files = sorted(os.listdir(images_dir))
            for filename in all_files:
                if not filename.lower().endswith(('.jpg', '.png')): continue
                
                # Check mask
                mask_path = os.path.join(masks_dir, filename.replace('.jpg', '.png'))
                if not os.path.exists(mask_path): mask_path = os.path.join(masks_dir, filename)
                
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    # KEEP IF HEALTHY (Sum == 0)
                    if mask is not None and np.sum(mask) == 0:
                        self.image_files.append(filename)

    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, img

class CrackTestDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])

    def __len__(self): return len(self.image_files)
    
    def __getitem__(self, idx):
        name = self.image_files[idx]
        img = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        mask_path = os.path.join(self.masks_dir, name.replace('.jpg','.png'))
        if not os.path.exists(mask_path): mask_path = os.path.join(self.masks_dir, name)
        
        if os.path.exists(mask_path): mask = Image.open(mask_path).convert('L')
        else: mask = Image.new('L', img.size, 0)
        
        if self.transform:
            img = self.transform(img)
            mask = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])(mask)
        return img, (mask > 0.5).float(), name

def get_loader(img_path, mask_path, batch_size, is_train=True):
    # Simple 0-1 Normalization (Matches Sigmoid Output)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor() 
    ])
    
    if is_train:
        ds = CrackTrainDataset(img_path, mask_path, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        ds = CrackTestDataset(img_path, mask_path, transform)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
