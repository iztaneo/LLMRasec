import random
import torch
from PIL import Image, ImageDraw, ImageFont

class SynthTextGenerator:
    """
    Generador Sintético de Imágenes de Texto con Fuentes Tipográficas Reales (PIL/Pillow).
    Genera tiras de imágenes de texto [B, 1, 32, 128] y secuencias de etiquetas para vocabulario completo (A-Z, a-z, 0-9).
    """
    def __init__(self, vocab=None, img_width=128, img_height=32):
        self.vocab = vocab or "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 "
        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}
        self.img_width = img_width
        self.img_height = img_height

        # Intentar cargar fuentes tipográficas del sistema
        self.fonts = []
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
        ]
        for path in font_paths:
            try:
                self.fonts.append(ImageFont.truetype(path, 18))
            except Exception:
                pass
        if not self.fonts:
            self.fonts.append(ImageFont.load_default())

    def generate_single_image(self, text):
        img = Image.new("L", (self.img_width, self.img_height), color=255)
        draw = ImageDraw.Draw(img)
        font = random.choice(self.fonts)

        # Centrar texto en la imagen
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = max(0, (self.img_width - text_w) // 2)
        y = max(0, (self.img_height - text_h) // 2)

        draw.text((x, y), text, fill=0, font=font)

        # Convertir a tensor PyTorch [1, H, W] normalizado de 0.0 a 1.0
        tensor_img = torch.tensor(list(img.getdata()), dtype=torch.float32).view(1, self.img_height, self.img_width) / 255.0
        return tensor_img

    def generate_batch(self, batch_size=8, texts=None):
        sample_words = [
            "MITSUBISHI", "MOTORS", "Toyota", "Honda", "Nissan",
            "Engine", "Speed", "Drive", "Japan", "Car",
            "ABCXYZ", "abcdef", "123456", "Hello", "World"
        ]
        if texts is None:
            texts = [random.choice(sample_words) for _ in range(batch_size)]

        img_list = []
        for text in texts:
            img_list.append(self.generate_single_image(text))

        images = torch.stack(img_list) # [B, 1, 32, 128]
        return images, texts
