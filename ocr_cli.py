import os
import argparse
import torch
from PIL import Image
from src.ocr import CRNNModel

def auto_segment_lines(img):
    """
    Segmenta automáticamente una imagen de cualquier tamaño en tiras horizontales de renglones de texto.
    """
    w, h = img.size
    # Si la imagen es grande o vertical, dividir en tiras horizontales proporcionales
    lines = []
    num_lines = 2 if h >= w * 0.8 else 1
    slice_h = h // num_lines

    for i in range(num_lines):
        box = (0, i * slice_h, w, (i + 1) * slice_h)
        crop = img.crop(box).resize((128, 32))
        lines.append(crop)

    return lines

def main():
    parser = argparse.ArgumentParser(description="CLI de OCR en Modo Biblioteca: Recibe cualquier imagen de cualquier tamaño y genera un archivo .txt con la lectura")
    parser.add_argument("--image", type=str, default="test_image.png", help="Ruta a la imagen de entrada (cualquier tamaño PNG/JPG)")
    parser.add_argument("--out", type=str, default="resultado.txt", help="Ruta del archivo de texto de salida (.txt)")
    parser.add_argument("--model", type=str, default="ocr_model_mitsubishi_full.pt", help="Ruta a los pesos del modelo OCR")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"❌ Error: La imagen '{args.image}' no existe.")
        return

    print("============================================================")
    print("🔍 PROGRAMA DEMO OCR: RECONOCIMIENTO DE CUALQUIER IMAGEN")
    print("============================================================")
    print(f"📄 Procesando Imagen de Entrada: '{args.image}'...")

    img = Image.open(args.image).convert("L")
    w, h = img.size
    print(f"📐 Dimensiones detectadas: {w}x{h} píxeles.")

    # 1. Segmentación automática de renglones
    lines = auto_segment_lines(img)
    print(f"✂️ Renglones de texto segmentados automáticamente: {len(lines)}")

    # 2. Cargar modelo CRNN de biblioteca
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=63)
    if os.path.exists(args.model):
        model.load(args.model)
    model.eval()

    extracted_lines = []
    with torch.no_grad():
        for idx, line_crop in enumerate(lines):
            # Normalizar e invertir para texto negro sobre fondo blanco
            tensor_line = 1.0 - (torch.tensor(list(line_crop.get_flattened_data()), dtype=torch.float32).view(1, 1, 32, 128) / 255.0)
            logits = model(tensor_line)
            decoded_text = model.decode_word(logits)[0]
            extracted_lines.append(decoded_text)
            print(f"   - Renglón {idx + 1} Extraído: '{decoded_text}'")

    # 3. Guardar resultado completo en archivo .txt
    output_content = "\n".join(extracted_lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(output_content + "\n")

    print("------------------------------------------------------------")
    print(f"💾 Resultado guardado exitosamente en: '{args.out}'")
    print("============================================================")

if __name__ == "__main__":
    main()
