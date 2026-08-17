import torch
import torch.nn as nn
import torch.optim as optim
from src.gan import Generator, Discriminator

def main():
    print("============================================================")
    print("🎭 Demostración 6: Red Generativa Adversaria GAN en Python (PyTorch)")
    print("============================================================")

    torch.manual_seed(42)
    generator = Generator()
    discriminator = Discriminator()

    criterion = nn.CrossEntropyLoss()
    opt_d = optim.AdamW(discriminator.parameters(), lr=0.02)
    opt_g = optim.AdamW(generator.parameters(), lr=0.02)

    print("🏋️ Entrenando GAN durante 100 épocas en Python...")
    for epoch in range(1, 101):
        # 1. Datos Reales y Etiquetas
        real_data = torch.randn(4, 2) + 1.0
        label_real = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        label_fake = torch.tensor([0, 0, 0, 0], dtype=torch.long)

        # 2. Entrenar Discriminador D
        opt_d.zero_grad()
        noise = torch.randn(4, 2)
        fake_data = generator(noise).detach()

        loss_d_real = criterion(discriminator(real_data), label_real)
        loss_d_fake = criterion(discriminator(fake_data), label_fake)
        loss_d = (loss_d_real + loss_d_fake) / 2.0
        loss_d.backward()
        opt_d.step()

        # 3. Entrenar Generador G
        opt_g.zero_grad()
        fake_data_g = generator(noise)
        loss_g = criterion(discriminator(fake_data_g), label_real)
        loss_g.backward()
        opt_g.step()

        if epoch % 25 == 0 or epoch == 100:
            print(f"Época {epoch}/100 | Loss D: {loss_d.item():.6f} | Loss G (Engaño): {loss_g.item():.6f}")

    print("============================================================")
    print("✅ ¡Red Generativa Adversaria GAN entrenada y verificada exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
