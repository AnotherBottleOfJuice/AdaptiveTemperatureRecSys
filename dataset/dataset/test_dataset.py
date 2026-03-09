from dataclasses import dataclass
import math

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class TestBatch:
    token_ids: torch.Tensor  # [B, T_max]
    lengths: torch.Tensor  # [B]


class TestDataset:
    def __init__(
            self,
            df: pl.DataFrame,
            batch_size: int,
            device: str = "cuda",
    ):
        self.df = df
        self.batch_size = batch_size
        self.device = device

        self._tensors = [torch.tensor(i) for i in df['token_id']]

    def __len__(self):
        return math.ceil(len(self._tensors) / self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            seqs = self._tensors[i * self.batch_size:
                        min((i + 1) * self.batch_size, len(self._tensors))]

            batch = pad_sequence(
                seqs,
                batch_first=True
            )
            lengths = torch.tensor([len(j) for j in seqs])
            yield TestBatch(batch, lengths)