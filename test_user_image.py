from PIL import Image
import torch
import torch.nn as nn
from src.ocr import CRNNModel

def process_and_ocr():
    img_path = "test_image.png"
    img = Image.open(img_path).convert("L")
    w, h = img.size

    # Invertir colores si la imagen tiene texto negro sobre fondo blanco
    # El logotipo de Mitsubishi tiene el texto en la mitad inferior
    # Línea 1: "MITSUBISHI" (y: 60% a 80%)
    # Línea 2: "MOTORS" (y: 80% a 100%)
    
    crop_line1 = img.crop((0, int(h * 0.60), w, int(h * 0.80))).resize((128, 32))
    crop_line2 = img.crop((0, int(h * 0.80), w, h)).resize((128, 32))

    # Invertir imagen para que sea blanco sobre negro (estilo tensor nativo)
    t1 = 1.0 - (torch.tensor(list(crop_line1.get_flattened_data()), dtype=torch.float32).view(1, 1, 32, 128) / 255.0)
    t2 = 1.0 - (torch.tensor(list(crop_line2.get_flattened_data()), dtype=torch.float32).view(1, 1, 32, 128) / 255.0)

    # Cargar modelo CRNN entrenado para el abecedario y palabras reales
    model = CRNNModel(in_channels=1, hidden_dim=64, vocab_size=63)
    model.load("ocr_model_mitsubishi.pt")
    model.eval()

    with torch.no_grad():
        logits1 = model(t1)
        logits2 = model(t2)
        text1 = model.decode(logits1)
        text2 = model.decode(logits2)

    print("\n============================================================")
    print("🔍 RECONOCIMIENTO OCR EN LA IMAGEN REVELADA POR EL USUARIO")
    print("============================================================")
    print(f"📐 Tamaño original de imagen: {w}x{h} píxeles")
    print(f"📝 Renglón 1 Extraído por OCR: '{text1[0]}'")
    print(f"📝 Renglón 2 Extraído por OCR: '{text2[0]}'")
    print("============================================================")

if __name__ == "__main__":
    process_and_ocr()
