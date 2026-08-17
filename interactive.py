import argparse
import torch
from src.utils import load_checkpoint, generate_text

def main():
    parser = argparse.ArgumentParser(description="Modo Interactivo CLI para chatear con tu LLM")
    parser.add_argument("--checkpoint", type=str, default="out/model.pt", help="Ruta al checkpoint .pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    print("=" * 60)
    print("💬 MODO INTERACTIVO DE LLM (Escribe 'salir' o 'exit' para terminar)")
    print("=" * 60)
    
    try:
        model, tokenizer, config = load_checkpoint(args.checkpoint, device=device)
        print(f"✅ Modelo cargado correctamente en [{device.upper()}].")
    except Exception as e:
        print(f"❌ Error cargando checkpoint desde '{args.checkpoint}': {e}")
        print("💡 Ejecuta 'python train.py' primero para entrenar tu modelo.")
        return

    temp = 0.7
    top_k = 40

    while True:
        try:
            prompt = input("\n👤 Tú: ")
            if prompt.strip().lower() in ["exit", "salir", "quit"]:
                print("👋 ¡Hasta luego!")
                break
            if not prompt.strip():
                continue

            print("🤖 LLM respondiendo...", end="", flush=True)
            output = generate_text(
                model, tokenizer, prompt=prompt, max_new_tokens=150, temperature=temp, top_k=top_k, device=device
            )
            print("\r" + " " * 30 + "\r🤖 LLM:\n" + output)

        except KeyboardInterrupt:
            print("\n👋 Sesión finalizada.")
            break

if __name__ == "__main__":
    main()
