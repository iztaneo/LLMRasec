import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class GPTConfig:
    """Configuración de hiperparámetros para la arquitectura GPT (Decoder-Only Transformer)."""
    vocab_size: int = 65
    block_size: int = 128     # Longitud de contexto máxima (T)
    n_layer: int = 4          # Número de capas Transformer
    n_head: int = 4           # Número de cabezas de atención
    n_embd: int = 128         # Dimensión de las incrustaciones (Embedding Dimension)
    dropout: float = 0.1      # Probabilidad de Dropout
    bias: bool = True         # Si se usa bias en las capas lineales


class CausalSelfAttention(nn.Module):
    """
    Módulo de Atención Causal Multi-Cabeza (Multi-Head Causal Self-Attention).
    Garantiza que un token solo pueda atender a tokens pasados y al actual (triangular mask).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd debe ser divisible por n_head"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Proyección combinada para Query, Key, Value
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Proyección de salida
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Máscara causal: buffer persistente que no se entrena (triangular inferior)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # Batch size, Sequence length, Embedding size (n_embd)

        # Calcular Q, K, V para todas las cabezas en un solo paso
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape a (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Uso de scaled_dot_product_attention si está disponible en PyTorch 2.0+ con máscara causal eficiente
        if hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout if self.training else 0.0, 
                is_causal=True
            )
        else:
            # Implementación manual transparente de Self-Attention Causal
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim)) # (B, nh, T, T)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, head_dim)

        # Re-ensamblar todas las cabezas en una sola dimensión de salida (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Proyección de salida
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Red Feed-Forward (MLP) de 2 capas lineales con activación GELU.
    Expande la dimensión interna a 4 * n_embd y luego proyecta de vuelta.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """
    Un Bloque de Transformer individual.
    Utiliza arquitectura Pre-LN (LayerNorm antes de Attention y MLP) para mayor estabilidad de gradiente.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """
    Modelo Completo de Lenguaje Autorregresivo (Decoder-Only Transformer).
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        
        # Cabeza del modelo de lenguaje para predecir sobre el vocabulario
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight Tying: Compartir pesos entre Embedding y LM Head (Estilo GPT-2)
        self.transformer.wte.weight = self.lm_head.weight

        # Inicialización de pesos según estándar de GPT
        self.apply(self._init_weights)

        # Ajuste especial de escalado para capas de proyección de salida residual
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Secuencia de longitud {t} excede la longitud de contexto de {self.config.block_size}"

        # Posiciones de los tokens [0, 1, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Incrustaciones de tokens y posición
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Pasar por cada bloque de Transformer
        for block in self.transformer.h:
            x = block(x)

        # LayerNorm final
        x = self.transformer.ln_f(x)

        if targets is not None:
            # Modo entrenamiento: calcular logits para todos los tokens y pérdida Cross Entropy
            logits = self.lm_head(x) # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Modo inferencia: optimizar calculando únicamente el logit del último token
            logits = self.lm_head(x[:, -1:, :]) # (B, 1, vocab_size)
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding=True):
        """Retorna el número total de parámetros entrenables en el modelo."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
