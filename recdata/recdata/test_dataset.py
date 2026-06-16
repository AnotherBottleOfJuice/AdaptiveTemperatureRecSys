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
        device: str,
    ):
        self.df = df
        self.batch_size = batch_size
        self.device = device
        self.ddp = False

        self._tensors = [torch.tensor(i) for i in df["token_id"]]

    def __len__(self):
        if self.ddp:
            return math.ceil(len(self._tensors) / (self.batch_size * self.world_size))
        return math.ceil(len(self._tensors) / self.batch_size)

    def ddp_setup(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self._tensors = self._tensors[
            : len(self._tensors) - (len(self._tensors) % (self.batch_size * world_size))
        ]
        self.ddp = True

    def __iter__(self):
        for i in range(len(self)):
            if self.ddp:
                seqs = self._tensors[
                    (i * self.world_size + self.rank) * self.batch_size : (
                        i * self.world_size + self.rank + 1
                    )
                    * self.batch_size,
                    len(self._tensors),
                ]
            else:
                seqs = self._tensors[i * self.batch_size : (i + 1) * self.batch_size]

            batch = pad_sequence(seqs, batch_first=True)
            lengths = torch.tensor([len(j) for j in seqs])
            yield TestBatch(batch, lengths)
