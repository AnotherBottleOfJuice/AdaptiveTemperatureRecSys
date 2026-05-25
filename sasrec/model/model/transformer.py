import torch
import torch.nn.functional as F
from torch import nn, Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()

        self.fc1 = torch.nn.Linear(d_model, 4 * d_model)
        self.fc2 = torch.nn.Linear(4 * d_model, d_model)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.dropout = torch.nn.Dropout(dropout)

        self.ln2 = torch.nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        dropout: float,
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
