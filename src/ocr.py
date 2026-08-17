import torch
import torch.nn as nn

class CRNNModel(nn.Module):
    """
    Modelo CRNN (Convolutional Recurrent Neural Network) para OCR de Renglones Completo.
    Extrae secuencias de palabras enteras (ej. "MITSUBISHI", "MOTORS") a lo largo de los timesteps horizontales.
    """
    def __init__(self, in_channels=1, hidden_dim=64, vocab_size=63):
        super().__init__()
        self.vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        self.vocab_size = len(self.vocab)

        # Extractor Visual CNN 2D
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [B, 16, 16, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [B, 32, 8, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((8, 1)) # [B, 64, 1, 32] (32 pasos temporales horizontales)
        )

        # Secuenciador BiLSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # Proyector a Vocabulario (63 caracteres)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        features = self.cnn(x).squeeze(2).permute(0, 2, 1) # [B, 32, 64]
        lstm_out, _ = self.lstm(features) # [B, 32, 128]
        logits = self.fc(lstm_out) # [B, 32, 63]
        return logits

    def decode_word(self, logits):
        """
        Decodifica la secuencia horizontal completa de caracteres a lo largo de los 32 timesteps
        para formar la palabra completa (ej. "MITSUBISHI" o "MOTORS").
        """
        preds = torch.argmax(logits, dim=2) # [B, 32]
        decoded_words = []

        for b in range(preds.size(0)):
            char_sequence = []
            prev_idx = -1
            for t_idx in preds[b]:
                idx = t_idx.item()
                if idx != prev_idx: # Colapsar repeticiones continuas del mismo carácter
                    if idx < len(self.vocab):
                        char = self.vocab[idx]
                        if char != ' ':
                            char_sequence.append(char)
                    prev_idx = idx
            decoded_words.append("".join(char_sequence))

        return decoded_words

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
