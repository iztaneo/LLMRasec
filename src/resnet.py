import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Bloque Residual ResNet (Skip Connection y = ReLU(f(x) + x))
    Paridad exacta con NeuralSuite layers/residual.h
    """
    def __init__(self, channels=8):
        super().__init__()
        self.fx = nn.Linear(channels, channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Conexión de salto shortcut: y = ReLU(f(x) + x)
        return self.relu(self.fx(x) + x)
