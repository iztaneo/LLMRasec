import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

class TextDataset(Dataset):
    """
    Dataset PyTorch para secuencias de texto tokenizadas.
    Genera pares (x, y) donde y es la secuencia x desplazada un token a la derecha.
    """
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


def prepare_dataset(
    text: str,
    tokenizer,
    block_size: int,
    batch_size: int,
    split_ratio: float = 0.9,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Tokeniza el texto completo y crea DataLoaders de entrenamiento y validación.
    """
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded, dtype=torch.long)

    # Separación Train / Val
    n = int(split_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]

    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader
