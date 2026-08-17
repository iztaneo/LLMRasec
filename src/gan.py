import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generador GAN: Convierte ruido z (2D) a datos sintéticos (2D)"""
    def __init__(self, latent_dim=2, hidden_dim=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """Discriminador GAN: Clasifica muestras entre Reales (1) o Falsas (0)"""
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
