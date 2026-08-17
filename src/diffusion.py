import math
import torch
import torch.nn as nn

class DenoisingModel(nn.Module):
    """
    Modelo de Desruidificado de Difusión (DDPM Toy PyTorch)
    Paridad exacta con NeuralSuite demo_diffusion.cpp
    """
    def __init__(self, dim=4, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, xt):
        return self.net(xt)
