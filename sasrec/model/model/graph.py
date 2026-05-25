import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from yambdadataset import TrainingBatch
from .transformer import GPT
from .tau import Tau, TauContext


class Graph(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        tau: Tau,
        n_layers: int,
        d_model: int,
        n_heads: int,
        dropout: float,
        log_q_correction: float,
        is_cosine_similarity: bool,
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
        self.tau = tau
        self.log_q_correction = log_q_correction
        self.register_buffer(
            "tokens_passed", torch.zeros((), dtype=torch.float32), persistent=False
        )
        self.is_cosine_similarity = is_cosine_similarity

    def forward(self, batch: TrainingBatch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            x: torch.Tensor = self.gpt(batch.inputs)  # (B, S, h)
            pos_weights = self.head.weight[batch.targets]  # (B, S, h)
            neg_weights = self.head.weight[batch.negatives]  # (N, h)

            if self.is_cosine_similarity:
                x = F.normalize(x, dim=-1)
                pos_weights = F.normalize(pos_weights, dim=-1)
                neg_weights = F.normalize(neg_weights, dim=-1)

            pos_logits = torch.sum(x * pos_weights, dim=2)  # (B, S)
            neg_logits = torch.matmul(x, neg_weights.T)  # (B, S, N)

            world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.tokens_passed.add_(float(batch.size * world_size))

            logits = torch.concat([pos_logits.unsqueeze(2), neg_logits], dim=-1)
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):

            tau = self.tau(TauContext(self, pos_logits, neg_logits, batch))
            tau = tau.to(device=logits.device, dtype=torch.float32)
            logits = logits.to(dtype=torch.float32)

            scaled_logits = logits / tau

            if (
                self.log_q_correction > 0.0
                and batch.positive_log_q is not None
                and batch.negative_log_q is not None
            ):
                pos_log_q = batch.positive_log_q.to(scaled_logits.dtype).unsqueeze(2)
                neg_log_q = batch.negative_log_q.to(scaled_logits.dtype).view(1, 1, -1)
                log_q = torch.concat(
                    [
                        pos_log_q,
                        neg_log_q.expand(scaled_logits.size(0), scaled_logits.size(1), -1),
                    ],
                    dim=-1,
                )
                scaled_logits = scaled_logits - self.log_q_correction * log_q

            loss = F.cross_entropy(
                scaled_logits.view(-1, scaled_logits.size(-1)),
                torch.zeros(
                    logits.size(0) * logits.size(1), device=logits.device, dtype=torch.long
                ),
            )

            metrics = {
                "train/pos_logit_mean": pos_logits.mean().item(),
                "train/neg_logit_mean": neg_logits.mean().item(),
                "train/tau": tau.mean().item(),
                "train/scaled_logit_mean": scaled_logits.mean().item(),
            }

            return loss, metrics
