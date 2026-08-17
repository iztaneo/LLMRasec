import torch
import torch.nn as nn

class OCRPipeline(nn.Module):
    """
    Pipeline OCR (Optical Character Recognition - CNN Visual Feature Extractor + Linear Classifier)
    Paridad exacta con NeuralSuite demo_ocr.cpp
    """
    def __init__(self, in_channels=1, num_classes=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 4, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(4 * 3 * 3, num_classes)

    def forward(self, x):
        h = self.conv(x)
        h = self.relu(h)
        h = self.pool(h)
        h_flat = h.view(x.size(0), -1)
        logits = self.classifier(h_flat)
        return logits
