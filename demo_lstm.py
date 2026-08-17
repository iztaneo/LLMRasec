import torch
import torch.nn as nn
import torch.optim as optim
from src.lstm import LSTMModel

def main():
    print("============================================================")
    print("🔄 Demostración 3: Red Recurrente (LSTM PyTorch)")
    print("============================================================")

    torch.manual_seed(42)
    # Tensor 3D de secuencias: [seq_len=5, batch_size=1, features=4]
    X = torch.randn(5, 1, 4)
    Y = torch.zeros(5, 1, 2)

    model = LSTMModel(input_dim=4, hidden_dim=8, output_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    print("🏋️ Entrenando LSTM durante 10 iteraciones en Python...")
    for epoch in range(1, 11):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        print(f"Época {epoch}/10 | Loss Recurrente LSTM: {loss.item():.6f}")

    print("============================================================")
    print("✅ ¡Entrenamiento LSTM completado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
