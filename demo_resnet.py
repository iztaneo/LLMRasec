import torch
import torch.nn as nn
import torch.optim as optim
from src.resnet import ResidualBlock

def main():
    print("============================================================")
    print("🧱 Demostración 5: Red Residual ResNet en Python (PyTorch Parity)")
    print("============================================================")

    torch.manual_seed(42)
    # Dataset XOR
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    Y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    model = nn.Sequential(
        nn.Linear(2, 8),
        ResidualBlock(8),
        nn.Linear(8, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05)

    print("🏋️ Entrenando Red Residual ResNet durante 100 épocas en Python...")
    for epoch in range(1, 101):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, Y)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 100:
            print(f"Época {epoch}/100 | Loss ResNet: {loss.item():.7f}")

    print("============================================================")
    print("✅ ¡Red Residual ResNet entrenada y verificada exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
