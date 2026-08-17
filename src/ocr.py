import torch
import torch.nn as nn

class CRNNModel(nn.Module):
    """
    Modelo CRNN (Convolutional Recurrent Neural Network) para OCR de Renglones de Texto completos.
    Conv2D (Extractor Visual 2D) + BiLSTM (Secuenciador) + Linear (Vocabulario Completo).
    """
    def __init__(self, in_channels=1, hidden_dim=64, vocab_size=63):
        super().__init__()
        self.vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        self.vocab_size = len(self.vocab)

        # 1. Extractor Visual 2D (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [B, 16, 16, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # [B, 32, 8, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((8, 1)) # Reducir alto a 1: [B, 64, 1, 32]
        )

        # 2. Secuenciador Recurrente Bidireccional (BiLSTM)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim, batch_first=True, bidirectional=True)

        # 3. Proyector a Vocabulario (A-Z, a-z, 0-9, espacio)
        self.fc = nn.Linear(hidden_dim * 2, self.vocab_size)

    def forward(self, x):
        # x: [B, 1, 32, 128]
        features = self.cnn(x) # [B, 64, 1, 32]
        features = features.squeeze(2) # [B, 64, 32]
        features = features.permute(0, 2, 1) # [B, 32, 64] (batch, timesteps, features)

        lstm_out, _ = self.lstm(features) # [B, 32, 128]
        logits = self.fc(lstm_out) # [B, 32, vocab_size]
        return logits

    def decode(self, logits):
        """
        Decodificación Codiciosa (Greedy Search) de la secuencia predicha.
        Colapsa caracteres repetidos consecutivos y elimina espacios vacíos extras.
        """
        preds = torch.argmax(logits, dim=2) # [B, timesteps]
        decoded_strings = []

        for b in range(preds.size(0)):
            raw_pred = preds[b]
            char_list = []
            prev_idx = -1
            for idx_tensor in raw_pred:
                idx = idx_tensor.item()
                if idx != prev_idx and idx < len(self.vocab):
                    char_list.append(self.vocab[idx])
                    prev_idx = idx
            text = "".join(char_list).strip()
            decoded_strings.append(text)

        return decoded_strings

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
