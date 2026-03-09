import torch
from torch import nn
import torch.nn.functional as F

from dataset import TrainingBatch
from .gpt import GPT

class Graph(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            n_layers: int = 4,
            d_model: int = 256,
            n_heads: int = 4,
            dropout: float = 0.0
    ):
        super().__init__()
        self.gpt = GPT(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, batch: TrainingBatch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x = self.gpt(batch.inputs)
            logits = self.head(x)

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            batch.targets.view(-1)
        )
