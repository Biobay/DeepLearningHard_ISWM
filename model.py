import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=2048):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 128x128
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 64x64
            
            nn.Conv2d(32, 64, 3, padding=1), # 64x64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),              # 32x32
            
            nn.Conv2d(64, 128, 3, padding=1),# 32x32
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)               # 16x16
        )
        
        # Flatten: 128 * 16 * 16 = 32768
        self.flatten_dim = 128 * 16 * 16
        
        # Bottleneck
        self.fc1 = nn.Linear(self.flatten_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, self.flatten_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 2, stride=2),   # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 2, stride=2),    # 128x128
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 16, 16)
        x = self.decoder(x)
        return x

def get_model(device, latent_dim=2048, image_size=128):
    model = ConvAutoencoder(latent_dim=latent_dim).to(device)
    return model
