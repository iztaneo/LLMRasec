import torch
import torch.nn as nn

class CRNNModel(nn.Module):
    """
    Modelo OCR en Modo Biblioteca PyTorch (CRNNModel)
    Paridad exacta con NeuralSuite include/models/ocr.h
    """
    def __init__(self, in_channels=1, hidden_dim=16, num_classes=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(4 * 3 * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(h)
        h = self.pool(h)
        h_flat = h.view(x.size(0), -1)
        h_fc1 = self.relu(self.fc1(h_flat))
        logits = self.fc2(h_fc1)
        return logits

    def decode(self, logits, vocab):
        preds = torch.argmax(logits, dim=1)
        return ''.join([vocab[p.item()] for p in preds])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
