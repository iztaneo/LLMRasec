import torch
import torch.nn as nn
import torch.optim as optim
from src.cnn import CNNModel

def main():
    print("============================================================")
    print("🖼️ Demostración 2: Red Convolucional (CNN 2D PyTorch)")
    print("============================================================")

    torch.manual_seed(42)
    # Tensor 4D de imágenes ficticias: [batch_size=2, channels=1, height=8, width=8]
    X = torch.randn(2, 1, 8, 8)
    Y = torch.tensor([0, 1], dtype=torch.long)

    model = CNNModel(in_channels=1, out_channels=4, kernel_size=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    print("🏋️ Entrenando CNN durante 20 iteraciones en Python...")
    for epoch in range(1, 21):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0 or epoch == 20:
            print(f"Época {epoch}/20 | Loss Convolucional: {loss.item():.6f}")

    print("============================================================")
    print("✅ ¡Entrenamiento CNN completado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
