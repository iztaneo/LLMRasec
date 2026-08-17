import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    Red Neuronal Convolucional 2D (Conv2D + MaxPool2D)
    Paridad exacta con NeuralSuite demo_cnn.cpp
    """
    def __init__(self, in_channels=1, out_channels=4, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(out_channels * 3 * 3, 2)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(h)
        h = self.pool(h)
        h_flat = h.view(x.size(0), -1)
        return self.fc(h_flat)
