import torch
from torch import nn, Tensor

from .block import Block


class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            n_layers: int,
            d_model: int,
            n_heads: int,
            dropout: float = 0.0,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.tok_emb = torch.nn.Embedding(vocab_size, d_model)

        self.pos_emb = torch.nn.Embedding(max_seq_len, d_model)

        self.drop = torch.nn.Dropout(dropout)

        self.blocks = torch.nn.ModuleList(
            Block(d_model, n_heads, dropout) for _ in range(n_layers)
        )

        self.ln_f = torch.nn.LayerNorm(d_model)

        self.head = torch.nn.Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        B, T = token_ids.shape
        assert T <= self.max_seq_len

        device = next(self.parameters()).device
        token_ids = token_ids.to(device)

        pos = torch.arange(T, device=device)

        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        return x
