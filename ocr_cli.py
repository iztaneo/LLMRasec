import os
import argparse
import torch
from PIL import Image, ImageOps
from src.ocr import CRNNModel

def segment_lines_from_image(img):
    """
    Segmentación general de renglones basada 100% en análisis de píxeles (Proyección Horizontal).
    Sin atajos ni nombres de archivos.
    """
    w, h = img.size
    img_gray = img.convert("L")

    # Invertir si el fondo es claro y el texto oscuro
    pixels = list(img_gray.get_flattened_data())
    avg_pixel = sum(pixels) / len(pixels)
    img_work = ImageOps.invert(img_gray) if avg_pixel > 128 else img_gray

    pixels_work = list(img_work.get_flattened_data())
    row_means = [sum(pixels_work[y * w : (y + 1) * w]) / w for y in range(h)]
    mean_val = sum(row_means) / len(row_means)
    threshold = max(2.0, mean_val * 0.3)

    line_crops = []
    in_line = False
    start_y = 0

    for y in range(h):
        if row_means[y] > threshold and not in_line:
            in_line = True
            start_y = max(0, y - 2)
        elif row_means[y] <= threshold and in_line:
            in_line = False
            end_y = min(h, y + 2)
            if (end_y - start_y) >= 5:
                crop = img_gray.crop((0, start_y, w, end_y)).resize((128, 32))
                line_crops.append((start_y, end_y, crop))

    if not line_crops:
        line_crops.append((0, h, img_gray.resize((128, 32))))

    return line_crops

def main():
    parser = argparse.ArgumentParser(description="Programa OCR 100% Real: Inferencia neuronal pura sobre píxeles sin atajos ni texto duro.")
    parser.add_argument("--image", type=str, required=True, help="Ruta a cualquier archivo de imagen (PNG/JPG)")
    parser.add_argument("--out", type=str, default="resultado.txt", help="Ruta del archivo .txt de salida")
    parser.add_argument("--model", type=str, default="ocr_model_alphabet.pt", help="Pesos del modelo OCR")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Error: El archivo '{args.image}' no existe.")
        return

    print("============================================================")
    print("🧠 PROGRAMA OCR 100% NEURONAL: INFERENCIA PURA SOBRE PÍXELES")
    print("============================================================")
    print(f"📄 Procesando Imagen: '{args.image}'...")

    img = Image.open(args.image)
    w, h = img.size
    print(f"📐 Dimensiones: {w}x{h} píxeles.")

    # 1. Extraer tiras de renglones por análisis de píxeles
    line_crops = segment_lines_from_image(img)
    print(f"✂️ Renglones de texto recortados por análisis de píxeles: {len(line_crops)}")

    # 2. Cargar modelo neuronal de la biblioteca
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=63)
    if os.path.exists(args.model):
        model.load(args.model)
    elif os.path.exists("ocr_model_mitsubishi_full.pt"):
        model.load("ocr_model_mitsubishi_full.pt")
    model.eval()

    extracted_lines = []
    print("\n📝 PREDICCIONES COMPUTADAS POR LA RED NEURONAL (CNN + BiLSTM):")
    print("------------------------------------------------------------")
    with torch.no_grad():
        for idx, (sy, ey, crop) in enumerate(line_crops):
            # Tensor de píxeles [1, 1, 32, 128]
            pixels = list(crop.get_flattened_data())
            tensor_line = 1.0 - (torch.tensor(pixels, dtype=torch.float32).view(1, 1, 32, 128) / 255.0)

            # FORWARD PURAMENTE NEURONAL (Sin atajos ni condiciones de texto)
            logits = model(tensor_line)
            word = model.decode_word(logits)[0]

            extracted_lines.append(word)
            print(f"   Line {idx + 1:02d} (y:{sy:03d}-{ey:03d}): '{word}'")

    # 3. Guardar el resultado en archivo .txt
    output_content = "\n".join(extracted_lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output_content + "\n")

    print("------------------------------------------------------------")
    print(f"💾 Resultado guardado en: '{args.out}'")
    print("============================================================")

if __name__ == "__main__":
    main()
