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
        with (torch.autocast(device_type="cuda", dtype=torch.bfloat16)):
            x: torch.Tensor = self.gpt(batch.inputs)  # (B, S, h)
            pos_weights = self.head.weight[batch.targets]  # (B, S, h)
            neg_weights = self.head.weight[batch.negatives]  # (N, h)
            pos_logist = torch.sum(x * pos_weights, dim=2)  # (B, S)
            neg_logits = torch.matmul(x, neg_weights.T)  # (B, S, N)
            logits = torch.concat((pos_logist.unsqueeze(2), neg_logits), dim=2)  # (B, S, N + 1)

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            torch.zeros(logits.size(0) * logits.size(1),
                        device=logits.device, dtype=torch.long),
        )
