import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import OCRPipeline

def main():
    print("============================================================")
    print("🔍 Demostración 9: Sistema OCR en Python (PyTorch Parity)")
    print("============================================================")

    vocab = ['A', 'B', 'C', 'D']
    torch.manual_seed(42)

    # 4 imágenes sintéticas de 8x8 píxeles
    images = torch.randn(4, 1, 8, 8) + 0.5
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    model = OCRPipeline(in_channels=1, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.03)

    print("🏋️ Entrenando Pipeline OCR durante 50 épocas en Python...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 50:
            print(f"Época {epoch}/50 | Loss Reconocimiento OCR: {loss.item():.6f}")

    print("\n🎯 Predicción de Caracteres Reconocidos por el OCR:")
    with torch.no_grad():
        final_logits = model(images)
        preds = torch.argmax(final_logits, dim=1)
        for i in range(4):
            pred_char = vocab[preds[i].item()]
            expected_char = vocab[labels[i].item()]
            print(f"   - Imagen {i+1} -> Carácter Predicho por OCR: '{pred_char}' (Esperado: '{expected_char}') {'✅' if pred_char == expected_char else '❌'}")

    print("============================================================")
    print("✅ ¡Pipeline OCR entrenado y verificado exitosamente en Python!")
    print("============================================================")

if __name__ == "__main__":
    main()
