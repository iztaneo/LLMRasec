import torch
import torch.nn as nn
import torch.optim as optim
from src.adaline import Adaline

def main():
    print("============================================================")
    print("🏛️ Demostración Histórica: ADALINE en Python (PyTorch Parity)")
    print("============================================================")

    torch.manual_seed(42)
    X = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    Y = torch.tensor([[-1.0], [-1.0], [-1.0], [1.0]])

    model = Adaline(input_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.05)

    print("🏋️ Entrenando ADALINE con la Regla Delta (LMS / Gradient Descent)...")
    for epoch in range(1, 101):
        optimizer.zero_grad()
        z = model(X)
        loss = criterion(z, Y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 100:
            print(f"Época {epoch}/100 | Pérdida LMS (MSE): {loss.item():.6f}")

    print("\n🎯 Predicciones de ADALINE tras la Función Umbral (Signo):")
    with torch.no_grad():
        z_final = model(X)
        for i in range(4):
            val_continua = z_final[i].item()
            pred_binaria = 1 if val_continua >= 0.0 else -1
            y_esperada = int(Y[i].item())
            print(f"   - Muestra [{X[i][0].item():.1f}, {X[i][1].item():.1f}] -> Salida Continua: {val_continua:.4f} | Predicción: {pred_binaria} (Esperado: {y_esperada}) {'✅' if pred_binaria == y_esperada else '❌'}")

    print("============================================================")
    print("✅ ¡Entrenamiento de ADALINE completado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
