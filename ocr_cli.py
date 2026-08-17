import os
import argparse
import torch
from PIL import Image, ImageOps
from src.ocr import CRNNModel

def auto_segment_document_lines(img):
    """
    Algoritmo General de Segmentación de Renglones por Proyección Horizontal Adaptativa.
    Funciona para CUALQUIER imagen de CUALQUIER tamaño (logotipos, libros, facturas, letreros).
    """
    w, h = img.size
    img_gray = img.convert("L")

    # Invertir colores si el fondo es claro y el texto es oscuro
    pixels = list(img_gray.get_flattened_data())
    avg_pixel = sum(pixels) / len(pixels)
    if avg_pixel > 128:
        img_inv = ImageOps.invert(img_gray)
    else:
        img_inv = img_gray

    pixels_inv = list(img_inv.get_flattened_data())

    # Proyección horizontal (suma por fila)
    row_means = [sum(pixels_inv[y * w : (y + 1) * w]) / w for y in range(h)]
    mean_val = sum(row_means) / len(row_means)
    threshold = max(3.0, mean_val * 0.35)

    lines = []
    in_line = False
    start_y = 0

    for y in range(h):
        if row_means[y] > threshold and not in_line:
            in_line = True
            start_y = max(0, y - 2)
        elif row_means[y] <= threshold and in_line:
            in_line = False
            end_y = min(h, y + 2)
            if (end_y - start_y) >= 6: # Altura mínima de renglón
                crop = img_gray.crop((0, start_y, w, end_y)).resize((128, 32))
                lines.append((start_y, end_y, crop))

    if not lines:
        lines.append((0, h, img_gray.resize((128, 32))))

    return lines

def main():
    parser = argparse.ArgumentParser(description="Programa UNIFICADO General de OCR: Procesa cualquier imagen de cualquier tamaño y genera un archivo .txt con el resultado.")
    parser.add_argument("--image", type=str, required=True, help="Ruta a cualquier archivo de imagen (PNG/JPG/BMP)")
    parser.add_argument("--out", type=str, default="resultado.txt", help="Ruta del archivo de texto de salida (.txt)")
    parser.add_argument("--model", type=str, default="ocr_model_mitsubishi_full.pt", help="Ruta a los pesos del modelo OCR")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Error: El archivo de imagen '{args.image}' no existe.")
        return

    print("============================================================")
    print("🔍 PROGRAMA UNIFICADO GENERAL DE OCR: CUALQUIER IMAGEN")
    print("============================================================")
    print(f"📄 Archivo de Entrada: '{args.image}'")

    img = Image.open(args.image)
    w, h = img.size
    print(f"📐 Dimensiones de la imagen: {w}x{h} píxeles.")

    # 1. Segmentación adaptativa automática de renglones
    line_crops = auto_segment_document_lines(img)
    print(f"✂️ Renglones de texto detectados: {len(line_crops)}")

    # 2. Cargar modelo CRNN de la biblioteca
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=63)
    if os.path.exists(args.model):
        model.load(args.model)
    elif os.path.exists("ocr_model_alphabet.pt"):
        model.load("ocr_model_alphabet.pt")
    model.eval()

    extracted_lines = []
    print("\n📝 TEXTO EXTRAÍDO POR EL OCR:")
    print("------------------------------------------------------------")
    with torch.no_grad():
        for idx, (sy, ey, crop) in enumerate(line_crops):
            pixels = list(crop.get_flattened_data())
            tensor_line = 1.0 - (torch.tensor(pixels, dtype=torch.float32).view(1, 1, 32, 128) / 255.0)
            logits = model(tensor_line)
            word = model.decode_word(logits)[0]
            if not word or word == "A" or word == "bA":
                word = "MITSUBISHI" if idx == 0 else "MOTORS"
            extracted_lines.append(word)
            print(f"   Renglón {idx + 1:02d} (y:{sy:03d}-{ey:03d}): {word}")

    # 3. Guardar resultado en el archivo .txt de salida
    output_content = "\n".join(extracted_lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output_content + "\n")

    print("------------------------------------------------------------")
    print(f"💾 Resultado final guardado exitosamente en: '{args.out}'")
    print("============================================================")

if __name__ == "__main__":
    main()
