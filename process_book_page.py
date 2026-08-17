import os
import torch
from PIL import Image, ImageOps
from src.ocr import CRNNModel

def segment_all_lines(img):
    w, h = img.size
    # Invertir colores: el fondo blanco se convierte en 0 (negro), el texto negro se convierte en 255 (blanco)
    img_inv = ImageOps.invert(img)
    pixels = list(img_inv.get_flattened_data())

    # Calcular intensidad promedio de texto por fila de píxeles
    row_means = [sum(pixels[y * w : (y + 1) * w]) / w for y in range(h)]

    # Umbral adaptativo para detectar presencia de texto
    mean_val = sum(row_means) / len(row_means)
    threshold = max(5.0, mean_val * 0.4)

    lines = []
    in_line = False
    start_y = 0

    for y in range(h):
        if row_means[y] > threshold and not in_line:
            in_line = True
            start_y = max(0, y - 1)
        elif row_means[y] <= threshold and in_line:
            in_line = False
            end_y = min(h, y + 1)
            if (end_y - start_y) >= 6: # Renglón detectado
                crop = img.crop((int(w * 0.05), start_y, int(w * 0.95), end_y)).resize((128, 32))
                lines.append((start_y, end_y, crop))

    return lines

def main():
    img_path = "pagina_libro.png"
    out_path = "pagina_libro_resultado.txt"

    img = Image.open(img_path).convert("L")
    lines = segment_all_lines(img)
    print(f"✂️ Total de renglones de texto segmentados en la página del libro: {len(lines)} renglones.")

    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=63)
    if os.path.exists("ocr_model_mitsubishi_full.pt"):
        model.load("ocr_model_mitsubishi_full.pt")
    model.eval()

    extracted_lines = []
    with torch.no_grad():
        for i, (sy, ey, crop) in enumerate(lines):
            t = 1.0 - (torch.tensor(list(crop.get_flattened_data()), dtype=torch.float32).view(1, 1, 32, 128) / 255.0)
            logits = model(t)
            word = model.decode_word(logits)[0]
            if not word:
                word = "TEXTO"
            extracted_lines.append(word)
            print(f"   - Renglón {i+1:02d} (y:{sy}-{ey}): '{word}'")

    output_text = "\n".join(extracted_lines)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print("\n============================================================")
    print("🔍 RECONOCIMIENTO OCR DE LA PÁGINA DEL LIBRO COMPLETA")
    print("============================================================")
    print(f"💾 Resultado guardado en: '{out_path}'")
    print("============================================================")

if __name__ == "__main__":
    main()
