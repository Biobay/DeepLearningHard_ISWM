"""
Convolutional Autoencoder per Anomaly Detection.

Architettura:
- Encoder: comprime l'immagine in una rappresentazione compatta (latent space)
- Decoder: ricostruisce l'immagine dal latent space

Principio: la rete impara a ricostruire immagini "normali" di asfalto.
Quando incontra una crepa (anomalia), la ricostruzione sarà imperfetta.
La differenza tra input e ricostruzione evidenzia l'anomalia.
"""

import torch
import torch.nn as nn

class ConvAutoencoder(nn.Module):
    """
    Autoencoder convoluzionale per image reconstruction.
    Supporta sia 128x128 che 256x256 automaticamente.
    """
    
    def __init__(self, latent_dim=128, image_size=128):
        """
        Args:
            latent_dim: dimensione del bottleneck (latent space)
            image_size: dimensione input (128 o 256)
        """
        super(ConvAutoencoder, self).__init__()
        
        self.image_size = image_size
        
        # ============ ENCODER ============
        # Comprime progressivamente l'immagine
        
        if image_size == 128:
            # 128x128 -> 4x4 (5 layer)
            self.encoder = nn.Sequential(
                # Layer 1: 128x128x3 -> 64x64x32
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Layer 2: 64x64x32 -> 32x32x64
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Layer 3: 32x32x64 -> 16x16x128
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Layer 4: 16x16x128 -> 8x8x256
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Layer 5: 8x8x256 -> 4x4x512
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.encoded_size = 4
            
        else:  # 256x256
            # 256x256 -> 8x8 (6 layer)
            self.encoder = nn.Sequential(
                # Layer 1: 256x256x3 -> 128x128x32
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Layer 2: 128x128x32 -> 64x64x64
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Layer 3: 64x64x64 -> 32x32x128
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Layer 4: 32x32x128 -> 16x16x256
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Layer 5: 16x16x256 -> 8x8x512
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                
                # Layer 6: 8x8x512 -> 8x8x512 (mantiene dimensione)
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.encoded_size = 8
        
        # ============ BOTTLENECK (Latent Space) ============
        
        self.flatten = nn.Flatten()
        self.bottleneck_size = 512 * self.encoded_size * self.encoded_size
        self.fc_encode = nn.Linear(self.bottleneck_size, latent_dim)
        
        # Decompressione dal latent space
        self.fc_decode = nn.Linear(latent_dim, self.bottleneck_size)
        self.unflatten = nn.Unflatten(1, (512, self.encoded_size, self.encoded_size))
        
        # ============ DECODER ============
        # Ricostruisce l'immagine (adattato per 128x128 o 256x256)
        
        if image_size == 128:
            # 4x4 -> 128x128 (5 layer)
            self.decoder = nn.Sequential(
                # Layer 1: 4x4x512 -> 8x8x256
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Layer 2: 8x8x256 -> 16x16x128
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Layer 3: 16x16x128 -> 32x32x64
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Layer 4: 32x32x64 -> 64x64x32
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Layer 5: 64x64x32 -> 128x128x3
                nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        else:  # 256x256
            # 8x8 -> 256x256 (5 layer)
            self.decoder = nn.Sequential(
                # Layer 1: 8x8x512 -> 16x16x256
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Layer 2: 16x16x256 -> 32x32x128
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Layer 3: 32x32x128 -> 64x64x64
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                # Layer 4: 64x64x64 -> 128x128x32
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                # Layer 5: 128x128x32 -> 256x256x3
                nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        """
        Forward pass: Encoder -> Bottleneck -> Decoder
        
        Args:
            x: input image [batch, 3, 128, 128]
        
        Returns:
            reconstructed: immagine ricostruita [batch, 3, 128, 128]
        """
        # Encoding: compressione
        encoded = self.encoder(x)
        
        # Bottleneck: rappresentazione compressa
        encoded_flat = self.flatten(encoded)
        latent = self.fc_encode(encoded_flat)
        
        # Decompressione dal latent space
        decoded_flat = self.fc_decode(latent)
        decoded = self.unflatten(decoded_flat)
        
        # Decoding: ricostruzione
        reconstructed = self.decoder(decoded)
        
        return reconstructed
    
    def get_latent_representation(self, x):
        """
        Estrae solo il latent vector (utile per analisi).
        """
        encoded = self.encoder(x)
        encoded_flat = self.flatten(encoded)
        latent = self.fc_encode(encoded_flat)
        return latent


def get_model(device='cuda', latent_dim=128, image_size=128):
    """
    Crea il modello e lo sposta sul device corretto.
    
    Args:
        device: 'cuda' o 'cpu'
        latent_dim: dimensione del bottleneck
        image_size: dimensione input (128 o 256)
    
    Returns:
        model: ConvAutoencoder
    """
    model = ConvAutoencoder(latent_dim=latent_dim, image_size=image_size)
    model = model.to(device)
    
    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Modello creato con {total_params:,} parametri ({trainable_params:,} trainable)")
    
    return model


if __name__ == "__main__":
    # Test del modello
    from config import IMAGE_SIZE
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = IMAGE_SIZE[0]  # Prendi la dimensione da config
    
    print(f"Device: {device}")
    print(f"Testing with image size: {img_size}x{img_size}")
    
    model = get_model(device=device, image_size=img_size)
    
    # Test con input dummy
    dummy_input = torch.randn(4, 3, img_size, img_size).to(device)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Shape matching: {dummy_input.shape == output.shape}")
