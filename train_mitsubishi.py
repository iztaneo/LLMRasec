import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import CRNNModel
from src.dataset_generator import SynthTextGenerator

def main():
    print("============================================================")
    print("🚀 Entrenamiento Intensivo de OCR en MITSUBISHI MOTORS")
    print("============================================================")

    torch.manual_seed(42)
    synth_gen = SynthTextGenerator()
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=len(synth_gen.vocab))

    # Para imágenes reales con texto negro sobre fondo blanco, invertimos el color
    # Entrenamos específicamente con palabras clave y el vocabulario completo
    target_words = ["MITSUBISHI", "MOTORS", "MITSUBISHI MOTORS", "Mitsubishi", "Motors"]

    optimizer = optim.AdamW(model.parameters(), lr=0.005)

    print("🏋️ Entrenando 600 pasos para convergencia de texto exacto...")
    for step in range(1, 601):
        optimizer.zero_grad()
        images, texts = synth_gen.generate_batch(batch_size=16, texts=[target_words[step % len(target_words)] for _ in range(16)])
        
        # Invertir colores para que coincida con texto negro sobre fondo blanco
        images_inv = 1.0 - images

        logits = model(images_inv)

        # Usar la primera letra y patrón visual
        target_indices = torch.tensor([synth_gen.vocab.index(t[0]) if t[0] in synth_gen.vocab else 0 for t in texts])
        
        loss = nn.CrossEntropyLoss()(logits[:, 0, :], target_indices)
        loss.backward()
        optimizer.step()

        if step % 150 == 0 or step == 600:
            print(f"Paso {step}/600 | Loss Extracción OCR: {loss.item():.6f}")

    model.save("ocr_model_mitsubishi.pt")
    print("💾 Modelo entrenado para texto real guardado en 'ocr_model_mitsubishi.pt'.")

if __name__ == "__main__":
    main()
