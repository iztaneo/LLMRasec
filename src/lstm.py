import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    Red Recurrente LSTM (Long Short-Term Memory)
    Paridad exacta con NeuralSuite demo_lstm.cpp
    """
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_seq, _ = self.lstm(x)
        logits_seq = self.fc(h_seq)
        return logits_seq
