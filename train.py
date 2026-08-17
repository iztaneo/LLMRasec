import os
import time
import math
import argparse
import torch
from src.tokenizer import CharTokenizer
from src.model import GPT, GPTConfig
from src.dataset import prepare_dataset
from src.utils import set_seed, generate_text, save_checkpoint

def get_args():
    parser = argparse.ArgumentParser(description="Entrenamiento de un LLM tipo GPT desde cero")
    parser.add_argument("--data_path", type=str, default="sample_data/input.txt", help="Ruta al archivo de texto de entrenamiento")
    parser.add_argument("--out_dir", type=str, default="out", help="Directorio para guardar los checkpoints")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de lote (batch size)")
    parser.add_argument("--block_size", type=int, default=64, help="Longitud del contexto (tokens por secuencia)")
    parser.add_argument("--n_layer", type=int, default=4, help="Número de capas de Transformer")
    parser.add_argument("--n_head", type=int, default=4, help="Número de cabezas de atención")
    parser.add_argument("--n_embd", type=int, default=128, help="Dimensión de embedding")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Tasa de aprendizaje (Learning rate)")
    parser.add_argument("--max_iters", type=int, default=600, help="Número máximo de iteraciones de entrenamiento")
    parser.add_argument("--eval_interval", type=int, default=100, help="Frecuencia de evaluación (iteraciones)")
    parser.add_argument("--eval_iters", type=int, default=20, help="Número de batches para promediar la pérdida en evaluación")
    parser.add_argument("--seed", type=int, default=1337, help="Semilla para reproducibilidad")
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_loader, val_loader, eval_iters, device):
    """Calcula la pérdida promedio (Cross-Entropy) en los conjuntos de entrenamiento y validación."""
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        loader_iter = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                try:
                    X, Y = next(loader_iter)
                except StopIteration:
                    break
            X, Y = X.to(device), Y.to(device)
            _, loss = model(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / max(1, len(losses))
    model.train()
    return out



def configure_optimizer(model, weight_decay, learning_rate, device_type):
    """Configura AdamW dividiendo parámetros 2D (matrices con decay) y 1D (biases/LayerNorm sin decay)."""
    decay_params = []
    nodecay_params = []

    # Para evitar procesar parámetros compartidos (weight tying) múltiples veces
    seen_params = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param in seen_params:
            continue
        seen_params.add(param)

        # Matriz de pesos 2D -> Weight Decay; Vectores 1D (biases, layernorm) -> No decay
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95))
    return optimizer



def get_lr(it: int, max_iters: int, learning_rate: float, warmup_iters: int = 50, min_lr: float = 1e-4) -> float:
    """Calcula el Learning Rate con Cosine Decay y Warmup."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= max_iters:
        return min_lr
    if max_iters <= warmup_iters:
        return learning_rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)



def main():
    args = get_args()
    set_seed(args.seed)

    # Selección automática del dispositivo de aceleración
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("=" * 60)
    print(f"🚀 Iniciando Entrenamiento del LLM en Dispositivo: {device.upper()}")
    print("=" * 60)

    # 1. Carga del conjunto de datos
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {args.data_path}")

    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"📄 Dataset cargado: {len(text):,} caracteres.")

    # 2. Inicialización del Tokenizador
    tokenizer = CharTokenizer(text)
    print(f"🔤 Vocabulario del tokenizador: {tokenizer.vocab_size} caracteres únicos.")

    # 3. Preparación de DataLoaders
    train_loader, val_loader = prepare_dataset(
        text, tokenizer, block_size=args.block_size, batch_size=args.batch_size
    )

    # 4. Construcción del Modelo GPT
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(config).to(device)
    num_params = model.get_num_params()
    print(f"🧠 Modelo GPT Creado | Parámetros entrenables: {num_params:,}")
    print("-" * 60)

    # 5. Configurar Optimizador
    optimizer = configure_optimizer(model, weight_decay=0.1, learning_rate=args.learning_rate, device_type=device)

    # 6. Bucle Principal de Entrenamiento
    model.train()
    start_time = time.time()
    train_iter = iter(train_loader)

    for iter_num in range(1, args.max_iters + 1):
        # Actualizar Tasa de Aprendizaje
        lr = get_lr(iter_num, args.max_iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Obtener siguiente lote de datos
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)

        X, Y = X.to(device), Y.to(device)

        # Paso Forward + Loss
        logits, loss = model(X, Y)

        # Paso Backward y Optimización
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluación periódica
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            losses = estimate_loss(model, train_loader, val_loader, args.eval_iters, device)
            elapsed = time.time() - start_time
            print(f"Step {iter_num:4d}/{args.max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | LR: {lr:.6f} | Tiempo: {elapsed:.2f}s")
            
            # Muestra de texto generado en tiempo real
            sample_text = generate_text(model, tokenizer, prompt="First Citizen:", max_new_tokens=80, temperature=0.7, device=device)
            print(f"  📝 [Texto Generado Iter {iter_num}]:\n  '{sample_text.strip()}'\n" + "-" * 60)

    # 7. Guardado final de Checkpoint y Vocabulario
    checkpoint_path = os.path.join(args.out_dir, "model.pt")
    save_checkpoint(model, optimizer, config, tokenizer, checkpoint_path)
    print("✅ ¡Entrenamiento completado exitosamente!")

if __name__ == "__main__":
    main()
