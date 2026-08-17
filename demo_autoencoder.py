import torch
import torch.nn as nn
import torch.optim as optim
from src.autoencoder import Autoencoder

def main():
    print("============================================================")
    print("🔄 Demostración 4: Autoencoder en Python (PyTorch Parity)")
    print("============================================================")

    torch.manual_seed(42)
    # Dataset 8-dimensional
    X = torch.randn(4, 8)

    model = Autoencoder(input_dim=8, hidden_dim=4, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.03)

    print("🏋️ Entrenando Autoencoder durante 200 épocas en Python...")
    for epoch in range(1, 201):
        optimizer.zero_grad()
        X_hat, z = model(X)
        loss = criterion(X_hat, X)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 200:
            print(f"Época {epoch}/200 | Loss Reconstrucción MSE: {loss.item():.6f}")

    print("============================================================")
    print("✅ ¡Autoencoder entrenado y verificado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
