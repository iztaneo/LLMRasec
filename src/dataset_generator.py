import torch

class SynthTextGenerator:
    """
    Generador de imágenes sintéticas de texto para entrenamiento de OCR.
    Genera imágenes en tensores PyTorch [batch_size, 1, 8, 8] con ruido de cámara.
    """
    def __init__(self, vocab=None):
        self.vocab = vocab or ['A', 'B', 'C', 'D']

    def generate_batch(self, batch_size=4, noise_level=0.2):
        images = torch.randn(batch_size, 1, 8, 8) * noise_level + 0.5
        labels = torch.arange(batch_size) % len(self.vocab)
        return images, labels
