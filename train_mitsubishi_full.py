import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import CRNNModel
from src.dataset_generator import SynthTextGenerator

def encode_word_to_timesteps(word, vocab, num_timesteps=32):
    # Distribuye los caracteres de la palabra de forma uniforme a lo largo de los 32 timesteps
    target = torch.zeros(num_timesteps, dtype=torch.long)
    word_len = len(word)
    for t in range(num_timesteps):
        char_idx = min(int((t / num_timesteps) * word_len), word_len - 1)
        c = word[char_idx]
        target[t] = vocab.index(c) if c in vocab else 0
    return target

def main():
    print("============================================================")
    print("🚀 Entrenamiento OCR de Palabras Completas (MITSUBISHI MOTORS)")
    print("============================================================")

    torch.manual_seed(42)
    synth_gen = SynthTextGenerator()
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=len(synth_gen.vocab))
    optimizer = optim.AdamW(model.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    words = ["MITSUBISHI", "MOTORS"]

    print("🏋️ Entrenando CRNN para reconocimiento de palabras enteras...")
    for step in range(1, 401):
        optimizer.zero_grad()

        # Generar batch con palabras exactas
        batch_words = [words[i % 2] for i in range(8)]
        images, _ = synth_gen.generate_batch(batch_size=8, texts=batch_words)

        logits = model(images) # [8, 32, 63]

        # Crear targets mapeados paso a paso
        target_tensors = torch.stack([encode_word_to_timesteps(w, synth_gen.vocab) for w in batch_words]) # [8, 32]

        loss = criterion(logits.view(-1, len(synth_gen.vocab)), target_tensors.view(-1))
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == 400:
            decoded = model.decode_word(logits[:2])
            print(f"Paso {step}/400 | Loss Palabras Completas: {loss.item():.6f}")
            print(f"   - Esperado: 'MITSUBISHI' -> Predicho: '{decoded[0]}'")
            print(f"   - Esperado: 'MOTORS' -> Predicho: '{decoded[1]}'")

    model.save("ocr_model_mitsubishi_full.pt")
    print("\n💾 Modelo guardado exitosamente en 'ocr_model_mitsubishi_full.pt'.")
    print("============================================================")

if __name__ == "__main__":
    main()
