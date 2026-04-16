from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from config import VOCAB_SIZE, UNIFORM_NEGATIVES_NUM, IN_BATCH_NEGATIVES_NUM, BOS


@dataclass
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor
    size: int
    negative_log_q: torch.Tensor | None = None
    positive_log_q: torch.Tensor | None = None


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
            uniform_negative_items: int = UNIFORM_NEGATIVES_NUM,
            in_batch_negative_items: int = IN_BATCH_NEGATIVES_NUM,
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
        self.uniform_negative_items = uniform_negative_items
        self.in_batch_negative_items = in_batch_negative_items
        self.min_freq = 1e-12

        self.batch_num_tokens = batch_size * seq_len + 1
        self.total_num_tokens = int(self.df.get_column("token_id").list.len().sum())

        uniform_prob = 1.0 / max(self.vocab_size - 1, 1)
        self.uniform_log_q = float(np.log(uniform_prob))

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def _in_batch_negative_log_q(self, batch_token_ids: torch.Tensor, ndx: torch.Tensor) -> torch.Tensor:
        batch_size = max(int(batch_token_ids.numel()), 1)
        sampled_token_ids = batch_token_ids.flatten()[ndx]
        multiplicity = torch.bincount(sampled_token_ids, minlength=self.vocab_size).to(self.device)
        q = multiplicity.to(dtype=torch.float32) / float(batch_size)
        q[BOS] = 1
        q = torch.clamp(q[sampled_token_ids], min=self.min_freq)
        return torch.log(q)

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
                if torch.cpu.is_available():
                    t_cpu = torch.as_tensor(buf[:self.batch_num_tokens], device='cpu')
                else:
                    t_cpu = torch.as_tensor(buf[:self.batch_num_tokens], device='mps')

                if self.pin_memory:
                    t_cpu = t_cpu.pin_memory()

                t = t_cpu.to(self.device)
                inputs = t[:-1].view((self.batch_size, self.seq_len))
                targets = t[1:].view((self.batch_size, self.seq_len))
                uniform_negatives = torch.randint(1, self.vocab_size, (self.uniform_negative_items,),
                                                  device=self.device)

                ndx = torch.randint(0, self.batch_num_tokens - 1, (self.in_batch_negative_items,), device=self.device)
                in_batch_negatives = t.flatten()[ndx]
                in_batch_mask = in_batch_negatives != BOS

                negatives = torch.concat([uniform_negatives, in_batch_negatives[in_batch_mask]])

                negative_log_q = torch.concat([
                    torch.full((uniform_negatives.numel(),), self.uniform_log_q, device=self.device),
                    self._in_batch_negative_log_q(t, ndx)[in_batch_mask],
                ])

                yield TrainingBatch(inputs=inputs,
                                    targets=targets,
                                    negatives=negatives,
                                    size=inputs.numel(),
                                    negative_log_q=negative_log_q,
                                    positive_log_q=None)
                buf = buf[self.batch_num_tokens:]
