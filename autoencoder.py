"""
Masked Denoising Autoencoder for Network Anomaly Detection
"""

import torch
import torch.nn as nn


class MaskedDenoisingAutoencoder(nn.Module):
    """Autoencoder with masking and denoising for network traffic analysis"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        
        # Encoder: compress to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Decoder: reconstruct from latent space
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x, mask):
        """Forward pass with noise injection and masking"""
        # Add noise to unmasked features only
        noise = torch.randn_like(x) * 0.05
        x_noisy = x + noise * mask

        # Encode and decode
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        
        return decoded


def masked_mse_loss(input, target, mask):
    """MSE loss computed only on unmasked features"""
    diff = (input - target) * mask
    loss = torch.sum(diff ** 2) / torch.sum(mask)
    return loss
