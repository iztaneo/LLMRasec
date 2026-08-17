import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import CRNNModel
from src.dataset_generator import SynthTextGenerator

def text_to_target_tensor(texts, vocab, max_len=32):
    char2idx = {c: i for i, c in enumerate(vocab)}
    batch_size = len(texts)
    targets = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        for j, c in enumerate(t[:max_len]):
            if c in char2idx:
                targets[i, j] = char2idx[c]
    return targets

def main():
    print("============================================================")
    print("🔤 Entrenamiento OCR Completo (A-Z, a-z, 0-9) con Fuentes Reales")
    print("============================================================")

    torch.manual_seed(42)
    synth_gen = SynthTextGenerator()
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=len(synth_gen.vocab))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003)

    print(f"🔤 Tamaño del Vocabulario: {len(synth_gen.vocab)} caracteres (Mayúsculas, Minúsculas, Números).")
    print("🏋️ Entrenando CRNN con generador sintético durante 200 iteraciones...")

    for step in range(1, 201):
        optimizer.zero_grad()
        images, texts = synth_gen.generate_batch(batch_size=16) # [16, 1, 32, 128]
        logits = model(images) # [16, 32, vocab_size]

        targets = text_to_target_tensor(texts, synth_gen.vocab, max_len=32)

        # Transponer para CrossEntropyLoss [B * timesteps, vocab_size]
        logits_flat = logits.view(-1, len(synth_gen.vocab))
        targets_flat = targets.view(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == 200:
            decoded = model.decode(logits[:2])
            print(f"Paso {step}/200 | Loss OCR: {loss.item():.6f}")
            print(f"   - Esperado 1: '{texts[0]}' -> Predicho: '{decoded[0]}'")
            print(f"   - Esperado 2: '{texts[1]}' -> Predicho: '{decoded[1]}'")

    model.save("ocr_model_alphabet.pt")
    print("\n💾 Modelo OCR entrenado guardado exitosamente en 'ocr_model_alphabet.pt'.")
    print("============================================================")

if __name__ == "__main__":
    main()
