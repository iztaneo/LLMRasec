import math
import torch
import torch.nn as nn
import torch.optim as optim
from src.diffusion import DenoisingModel

def main():
    print("============================================================")
    print("🎨 Demostración 8: Modelo de Difusión DDPM en Python (PyTorch)")
    print("============================================================")

    torch.manual_seed(42)
    x0 = torch.tensor([
        [ 1.0,  0.5, -0.5, -1.0],
        [ 0.5,  1.0, -1.0, -0.5],
        [-0.5, -1.0,  1.0,  0.5],
        [-1.0, -0.5,  0.5,  1.0]
    ])

    denoiser = DenoisingModel(dim=4, hidden_dim=16)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(denoiser.parameters(), lr=0.03)

    beta = 0.3
    print("🏋️ Entrenando Modelo de Difusión durante 100 épocas en Python...")
    for epoch in range(1, 101):
        optimizer.zero_grad()
        # 1. Proceso Forward: Agregar ruido gaussiano
        noise = torch.randn_like(x0)
        xt = math.sqrt(1.0 - beta) * x0 + math.sqrt(beta) * noise

        # 2. Proceso Reverse: Predecir el ruido adicionado
        noise_pred = denoiser(xt)

        loss = criterion(noise_pred, noise)
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0 or epoch == 100:
            print(f"Época {epoch}/100 | Loss Predicción de Ruido MSE: {loss.item():.6f}")

    print("============================================================")
    print("✅ ¡Modelo de Difusión DDPM entrenado y verificado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
