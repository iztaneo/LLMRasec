import torch
import torch.nn as nn
import torch.optim as optim
from src.ocr import CRNNModel
from src.dataset_generator import SynthTextGenerator

def main():
    print("============================================================")
    print("🔤 Entrenamiento General de OCR Universal (nn.CTCLoss)")
    print("============================================================")

    torch.manual_seed(42)
    synth_gen = SynthTextGenerator()
    vocab = synth_gen.vocab
    char2idx = {c: i + 1 for i, c in enumerate(vocab)} # Reservar 0 para el token de relleno (blank CTC)

    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=len(vocab) + 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    sample_vocabulary_words = [
        "MITSUBISHI", "MOTORS", "TEXTO", "LIGERAMENTE", "LIGERA",
        "adverbio", "singularidad", "Soldados", "armados", "armadura",
        "Toyota", "Honda", "Nissan", "Engine", "Speed", "Drive",
        "ABCXYZ", "abcdef", "123456", "Hello", "World"
    ]

    print(f"🔤 Tamaño del Vocabulario: {len(vocab)} caracteres (A-Z, a-z, 0-9).")
    print("🏋️ Entrenando CRNN con CTCLoss durante 800 iteraciones...")

    for step in range(1, 801):
        optimizer.zero_grad()
        # Generar palabras aleatorias del vocabulario real
        batch_words = [sample_vocabulary_words[step % len(sample_vocabulary_words)] for _ in range(8)]
        images, _ = synth_gen.generate_batch(batch_size=8, texts=batch_words)

        logits = model(images) # [8, 32, vocab_size + 1]
        log_probs = logits.log_softmax(2).permute(1, 0, 2) # [32, 8, vocab_size + 1] para CTCLoss

        # Construir targets tensoriales para CTCLoss
        targets_list = []
        target_lengths = []
        for w in batch_words:
            t = [char2idx[c] for c in w if c in char2idx]
            targets_list.extend(t)
            target_lengths.append(len(t))

        targets = torch.tensor(targets_list, dtype=torch.long)
        input_lengths = torch.full(size=(8,), fill_value=32, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == 800:
            decoded = model.decode_word(logits[:2])
            print(f"Paso {step}/800 | Loss CTC OCR: {loss.item():.6f}")
            print(f"   - Esperado 1: '{batch_words[0]}' -> Predicho: '{decoded[0]}'")
            print(f"   - Esperado 2: '{batch_words[1]}' -> Predicho: '{decoded[1]}'")

    model.save("ocr_model_alphabet.pt")
    print("\n💾 Modelo OCR Universal guardado exitosamente en 'ocr_model_alphabet.pt'.")
    print("============================================================")

if __name__ == "__main__":
    main()
