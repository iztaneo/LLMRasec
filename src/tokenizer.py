import json
import os
from typing import List, Dict, Union

class CharTokenizer:
    """
    Un tokenizador a nivel de caracteres simple pero completo para el LLM.
    Mapea cada carácter único en el conjunto de datos a un número entero.
    """
    def __init__(self, text: str = None):
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        
        if text is not None:
            self.build_vocab(text)
        else:
            self.chars: List[str] = []
            self.stoi: Dict[str, int] = {}
            self.itos: Dict[int, str] = {}
            self.vocab_size: int = 0

    def build_vocab(self, text: str):
        """Construye el vocabulario a partir del texto proporcionado."""
        unique_chars = sorted(list(set(text)))
        # Reservar tokens especiales si fueran necesarios, aunque para char-level usamos los del texto + UNK
        self.chars = [self.pad_token, self.unk_token] + unique_chars
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text: str) -> List[int]:
        """Convierte una cadena de texto en una lista de índices de tokens."""
        unk_idx = self.stoi.get(self.unk_token, 1)
        return [self.stoi.get(c, unk_idx) for c in text]

    def decode(self, indices: Union[List[int], List[List[int]]]) -> str:
        """Convierte una lista de índices de tokens de vuelta a texto."""
        if len(indices) > 0 and isinstance(indices[0], list):
            # En caso de recibir una lista de listas
            return [self.decode(sub) for sub in indices]
        
        chars = []
        for idx in indices:
            ch = self.itos.get(idx, self.unk_token)
            if ch not in (self.pad_token, self.unk_token):
                chars.append(ch)
            elif ch == self.unk_token:
                chars.append("░") # Representación visual para UNK
        return "".join(chars)

    def save(self, filepath: str):
        """Guarda el vocabulario en un archivo JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "chars": self.chars,
            "stoi": self.stoi,
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "CharTokenizer":
        """Carga un tokenizador guardado desde un archivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.chars = data["chars"]
        tokenizer.stoi = data["stoi"]
        tokenizer.itos = {int(k) if k.isdigit() else i: v for i, (k, v) in enumerate(data["stoi"].items())}
        # Asegurar reconversión int key
        tokenizer.itos = {v: k for k, v in tokenizer.stoi.items()}
        tokenizer.vocab_size = data["vocab_size"]
        return tokenizer

    def __len__(self):
        return self.vocab_size
