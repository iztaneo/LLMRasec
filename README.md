# 🧠 LLM desde Cero (Decoder-Only Transformer)

¡Bienvenido al proyecto **LLM-rasec**! Una implementación completa, didáctica y modular de un **Modelo de Lenguaje a gran escala (LLM) tipo GPT desde cero** usando **Python** y **PyTorch**.

---

## 📌 Arquitectura del Modelo

El modelo sigue la arquitectura **Decoder-Only Transformer** (similar a GPT-2 / GPT-3 / LLaMA):

1. **Tokenización a nivel de caracteres (o modular BPE)**: Mapea texto a índices de tokens y viceversa.
2. **Token & Positional Embeddings**: Combina representaciones semánticas de tokens ($W_{te}$) e información posicional ($W_{pe}$).
3. **Causal Multi-Head Self-Attention**:
   - Proyección lineal a Queries ($Q$), Keys ($K$) y Values ($V$).
   - Escalado por $\frac{1}{\sqrt{d_k}}$.
   - **Máscara Causal (Triangular)**: Garantiza que el token en la posición $t$ solo pueda atender a posiciones $\le t$.
   - **Softmax** y cálculo de salida residual.
4. **Feed-Forward Networks (MLP)**: Red lineal de 2 capas con activación **GELU**.
5. **Transformer Blocks con Pre-LayerNorm**: Estabilidad de gradiente mediante normalización previa a la atención y MLP.
6. **Weight Tying**: Compartición de pesos entre la matriz de Embedding de entrada y la cabeza de salida $LM\_Head$.

---

## 📁 Estructura del Proyecto

```
LLM-rasec/
├── requirements.txt         # Dependencias del proyecto (PyTorch, NumPy, TQDM)
├── sample_data/
│   └── input.txt            # Dataset de texto para entrenamiento (Shakespeare)
├── src/
│   ├── __init__.py
│   ├── tokenizer.py         # Tokenizador CharTokenizer con exportación/importación JSON
│   ├── model.py             # Módulos de PyTorch (GPT, Block, CausalSelfAttention, MLP, GPTConfig)
│   ├── dataset.py           # TextDataset y función prepare_dataset para DataLoaders (x, y shifted)
│   └── utils.py             # Muestreo autorregresivo (Temp, Top-K, Top-P) y guardar/cargar checkpoints
├── train.py                 # Script principal de entrenamiento con AdamW y Cosine Annealing
├── generate.py              # Inferencia de texto con un prompt desde el terminal
├── interactive.py           # Modo interactivo CLI estilo chat
└── README.md
```

---

## 🚀 Guía de Inicio Rápido

### 1. Requisitos Previos

Asegúrate de contar con el entorno de Python configurado:
```bash
./venv/bin/pip install -r requirements.txt
```

### 2. Entrenar el Modelo

Para iniciar el entrenamiento con los hiperparámetros por defecto sobre el dataset de ejemplo:
```bash
./venv/bin/python train.py --max_iters 500 --batch_size 32 --block_size 64
```

Opciones útiles de `train.py`:
- `--data_path`: Ruta a tu propio archivo de texto `.txt`.
- `--n_layer`: Número de bloques Transformer (ej. 4 u 8).
- `--n_head`: Número de cabezas de atención (ej. 4 u 8).
- `--n_embd`: Dimensión del embedding (ej. 128 o 256).
- `--learning_rate`: Tasa de aprendizaje inicial (default: `1e-3`).

### 3. Generar Texto

Una vez entrenado y guardado el checkpoint en `out/model.pt`:
```bash
./venv/bin/python generate.py --prompt "First Citizen:\n" --max_new_tokens 200 --temperature 0.7
```

### 4. Modo Interactivo (Chat CLI)

Para conversar interactivamente con tu LLM:
```bash
./venv/bin/python interactive.py
```

---

## 🔬 Verificación y Pruebas Unitarias

El proyecto incluye verificaciones de:
- Dimensiones de tensores en paso forward (`(B, T) -> (B, T, vocab_size)`).
- Pérdida Cross-Entropy esperada en un modelo sin entrenar ($\approx \ln(\text{vocab\_size})$).
- Muestreo con Temperatura, Top-K y Top-P (Nucleus Sampling).
