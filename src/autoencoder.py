import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder PyTorch (Encoder-Decoder Bottleneck Reconstruction)
    Paridad exacta con NeuralSuite demo_autoencoder.cpp
    """
    def __init__(self, input_dim=8, hidden_dim=4, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
