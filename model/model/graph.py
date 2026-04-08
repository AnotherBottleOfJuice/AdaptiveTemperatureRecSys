import torch
from torch import nn
import torch.nn.functional as F
from comet_ml import Experiment

from dataset import TrainingBatch
from config import LOG_Q_CORRECTION
from .gpt import GPT


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

    def forward(self, batch: TrainingBatch, writer: Experiment | None = None):
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
                writer.log_metric("train/pos_logit_mean", value=pos_logits.mean(), step=self.tokens_passed)
                writer.log_metric("train/neg_logit_mean", value=neg_logits.mean(), step=self.tokens_passed)
                writer.log_metric("train/neg_logit_max", value=neg_logits.max(), step=self.tokens_passed)
                writer.log_metric("train/pos_logit_min", value=pos_logits.min(), step=self.tokens_passed)

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
