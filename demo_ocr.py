import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import CRNNModel
from src.dataset_generator import SynthTextGenerator

def main():
    print("============================================================")
    print("🔍 Demostración 9: Sistema OCR en Modo Biblioteca Python")
    print("============================================================")

    vocab = ['A', 'B', 'C', 'D']
    torch.manual_seed(42)

    synth_gen = SynthTextGenerator(vocab=vocab)
    images, labels = synth_gen.generate_batch(batch_size=4)

    model = CRNNModel(in_channels=1, hidden_dim=16, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.03)

    print("🏋️ Entrenando CRNNModel de Biblioteca durante 50 épocas en Python...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 50:
            print(f"Época {epoch}/50 | Loss Reconocimiento CRNN: {loss.item():.6f}")

    # Guardar pesos con el método de biblioteca
    model.save("ocr_model.pt")
    print("💾 Pesos del modelo OCR guardados en 'ocr_model.pt'.")

    print("\n🎯 Predicción de Caracteres usando CRNNModel.decode de Biblioteca:")
    with torch.no_grad():
        final_logits = model(images)
        decoded_text = model.decode(final_logits, vocab)
        for i in range(4):
            expected_char = vocab[labels[i].item()]
            pred_char = decoded_text[i]
            print(f"   - Imagen {i+1} -> Carácter Decodificado: '{pred_char}' (Esperado: '{expected_char}') {'✅' if pred_char == expected_char else '❌'}")

    print("============================================================")
    print("✅ ¡Demostración de OCR en modo biblioteca completada exitosamente!")
    print("============================================================")

if __name__ == "__main__":
    main()
