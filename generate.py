import argparse
import torch
from src.utils import load_checkpoint, generate_text

def main():
    parser = argparse.ArgumentParser(description="Inferencia y generación de texto con el LLM entrenado")
    parser.add_argument("--checkpoint", type=str, default="out/model.pt", help="Ruta al archivo checkpoint .pt")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Texto de inicio (prompt) para la generación")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Número de tokens a generar")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperatura para controlar la aleatoriedad (0.0 a 1.5)")
    parser.add_argument("--top_k", type=int, default=40, help="Filtro Top-K para probabilidad de tokens")
    parser.add_argument("--top_p", type=float, default=0.9, help="Filtro Top-P (Nucleus Sampling)")
    args = parser.parse_args()

    # Selección de dispositivo
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"🔄 Cargando modelo desde: {args.checkpoint}...")
    model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
    print(f"✅ Modelo cargado ({model.get_num_params():,} parámetros).")

    print("\n" + "=" * 60)
    print(f"🤖 GENERACIÓN AUTORREGRESIVA (Temp: {args.temperature}, Top-K: {args.top_k}, Top-P: {args.top_p})")
    print("=" * 60)
    
    generated = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device
    )
    print(generated)
    print("=" * 60)

if __name__ == "__main__":
    main()
