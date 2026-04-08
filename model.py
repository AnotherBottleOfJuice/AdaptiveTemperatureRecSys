import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import TrainingBatch
from config import LOG_Q_CORRECTION

class MLP(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
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



class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
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
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.proj(y)
        return y


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
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

class Graph(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            max_seq_len: int,
            n_layers: int = 4,
            d_model: int = 256,
            n_heads: int = 4,
            dropout: float = 0.0,
            tau: float = 1.0,
            log_q_correction: float = LOG_Q_CORRECTION,
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
        self.head = nn.Embedding(vocab_size, d_model)
        self.tau = tau
        self.log_q_correction = log_q_correction
        self.tokens_passed = 0

    def forward(self, batch: TrainingBatch, writer: SummaryWriter | None = None):
        with ((torch.autocast(device_type="cuda", dtype=torch.bfloat16))):
            x: torch.Tensor = self.gpt(batch.inputs)  # (B, S, h)
            pos_weights = self.head(batch.targets)  # (B, S, h)
            neg_weights = self.head(batch.negatives)  # (N, h)

            x = F.normalize(x, dim=-1)
            pos_weights = F.normalize(pos_weights, dim=-1)
            neg_weights = F.normalize(neg_weights, dim=-1)

            pos_logits = torch.sum(x * pos_weights, dim=2)  # (B, S)
            neg_logits = torch.matmul(x, neg_weights.T)  # (B, S, N)

            if writer is not None:
                writer.add_scalar("train/pos_logit_mean", pos_logits.mean(), self.tokens_passed)
                writer.add_scalar("train/neg_logit_mean", neg_logits.mean(), self.tokens_passed)
                writer.add_scalar("train/neg_logit_max", neg_logits.max(), self.tokens_passed)
                writer.add_scalar("train/pos_logit_min", pos_logits.min(), self.tokens_passed)

            if self.log_q_correction > 0.0 and batch.positive_log_q is not None and batch.negative_log_q is not None:
                pos_logits = pos_logits - self.log_q_correction * batch.positive_log_q.to(pos_logits.dtype)
                neg_logits = (neg_logits -
                              self.log_q_correction * batch.negative_log_q.to(neg_logits.dtype).view(1, 1, -1))

            logits = torch.concat([
                pos_logits.unsqueeze(2),
                neg_logits
            ], dim=-1)

            self.tokens_passed += batch.size

        return -(pos_logits / self.tau - torch.logsumexp(logits / self.tau, dim=2)).mean()
