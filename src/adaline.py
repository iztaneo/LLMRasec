import torch
import torch.nn as nn

class Adaline(nn.Module):
    """
    ADALINE (ADAptive LINear Element - Widrow & Hoff, 1960)
    Paridad exacta con NeuralSuite demo_adaline.cpp
    """
    def __init__(self, input_dim=2):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
