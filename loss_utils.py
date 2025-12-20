import torch
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        # Load VGG16, pretrained on ImageNet
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        
        # Extract layers that capture texture (Relu1_2, Relu2_2, Relu3_3)
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        
        for x in range(4):
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg[x])
            
        # Freeze VGG (we don't train it)
        for param in self.parameters():
            param.requires_grad = False
            
        self.to(device)
        self.l1_loss = nn.L1Loss() # MAE

    def forward(self, recon, target):
        # 1. Pixel-wise MAE Loss (Structure)
        loss_mae = self.l1_loss(recon, target)
        
        # 2. Perceptual Loss (Texture)
        # Normalize for VGG (approximate)
        h_recon = (recon - 0.5) / 0.5
        h_target = (target - 0.5) / 0.5
        
        # Get features
        r1 = self.slice1(h_recon); t1 = self.slice1(h_target)
        r2 = self.slice2(r1);      t2 = self.slice2(t1)
        r3 = self.slice3(r2);      t3 = self.slice3(t2)
        
        loss_vgg = self.l1_loss(r1, t1) + self.l1_loss(r2, t2) + self.l1_loss(r3, t3)
        
        # Weighted Sum: 60% MAE, 40% VGG
        return 0.6 * loss_mae + 0.4 * (loss_vgg * 0.1)
