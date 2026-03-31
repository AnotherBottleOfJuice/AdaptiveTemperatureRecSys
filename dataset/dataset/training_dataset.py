from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from config import VOCAB_SIZE, NEGATIVE_ITEMS


@dataclass
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor
    size: int


class TrainingDataset:
    def __init__(
            self,
            df: pl.DataFrame,
            batch_size: int,
            seq_len: int,
            device: str = "cuda",
            chunk_rows: int = 64_000,
            shuffle: bool = True,
            seed: int | None = 42,
            pin_memory: bool = True,
            vocab_size: int = VOCAB_SIZE,
            negative_items: int = NEGATIVE_ITEMS,
    ):
        self.df = df
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.chunk_rows = chunk_rows
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size
        self.negative_items = negative_items

        self.batch_num_tokens = batch_size * seq_len + 1
        self.total_num_tokens = int(self.df.get_column("token_id").list.len().sum())

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def __iter__(self):
        df = self.df

        if self.shuffle:
            df = df.sample(fraction=1, shuffle=True, seed=self.seed)

        buf = np.ndarray((0,), dtype=np.int64)
        for i in range(0, df.height // self.chunk_rows + 1):
            chunk = df.slice(i * self.chunk_rows,
                             min(df.height - i * self.chunk_rows, self.chunk_rows))
            tokens = chunk.explode('token_id').get_column('token_id').cast(int).to_numpy()
            buf = np.concat((buf, tokens))
            while len(buf) >= self.batch_num_tokens + 1:
                t_cpu = torch.as_tensor(buf[:self.batch_num_tokens], device='cpu')
                if self.pin_memory:
                    t_cpu = t_cpu.pin_memory()
                t = t_cpu.to(self.device)
                inputs = t[:-1].view((self.batch_size, self.seq_len))
                targets = t[1:].view((self.batch_size, self.seq_len))
                negatives = torch.randint(1, self.vocab_size, (self.negative_items,), device=self.device)
                yield TrainingBatch(inputs=inputs, targets=targets,
                                    negatives=negatives, size=inputs.numel())
                buf = buf[self.batch_num_tokens:]
