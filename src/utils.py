import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

def set_seed(seed: int = 42):
    """Establece las semillas de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    device: str = "cpu"
) -> str:
    """
    Genera texto autorregresivo a partir de un prompt utilizando muestreo con Temperatura, Top-K y Top-P (Nucleus).
    """
    model.eval()
    
    # Si el prompt está vacío, comenzar con el token inicial (o token 0)
    if prompt:
        tokens = tokenizer.encode(prompt)
    else:
        tokens = [0]

    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0) # (1, T)
    block_size = model.config.block_size

    for _ in range(max_new_tokens):
        # Recortar el contexto si excede la longitud máxima del bloque
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Inferencia del modelo
        logits, _ = model(idx_cond) # (1, 1, vocab_size)
        logits = logits[:, -1, :] # Tomar el último paso temporal (1, vocab_size)

        # Aplicar Temperatura
        if temperature > 0:
            logits = logits / temperature
        else:
            # Temperatura 0 equivale a muestreo codicioso (Greedy decoding)
            _, next_token = torch.topk(logits, k=1, dim=-1)
            idx = torch.cat((idx, next_token), dim=1)
            continue

        # Aplicar Top-K Filtering
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # Aplicar Top-P (Nucleus) Filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Eliminar tokens con probabilidad acumulada mayor a top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Desplazar a la derecha para mantener al menos el primer token por encima del umbral
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')

        # Calcular probabilidades softmax y muestrear el siguiente token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1) # (1, 1)

        # Anexar el nuevo token a la secuencia generada
        idx = torch.cat((idx, next_token), dim=1)

    generated_tokens = idx[0].tolist()
    return tokenizer.decode(generated_tokens)


def save_checkpoint(model, optimizer, config, tokenizer, filepath: str):
    """Guarda un checkpoint del modelo entrenado y su tokenizador."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, filepath)
    tokenizer_path = os.path.join(os.path.dirname(filepath), "vocab.json")
    tokenizer.save(tokenizer_path)
    print(f"Checkpoint guardado exitosamente en: {filepath}")


def load_checkpoint(filepath: str, device: str = "cpu"):
    """Carga un modelo y su tokenizador desde un checkpoint guardado."""
    from .model import GPT
    from .tokenizer import CharTokenizer

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint["config"]
    
    tokenizer_path = os.path.join(os.path.dirname(filepath), "vocab.json")
    tokenizer = CharTokenizer.load(tokenizer_path)

    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, tokenizer, config
