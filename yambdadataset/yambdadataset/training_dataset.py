from dataclasses import dataclass

import polars as pl
import torch
from torch.utils.data import IterableDataset

BOS = 0


@dataclass
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor
    size: int
    negative_log_q: torch.Tensor | None
    positive_log_q: torch.Tensor | None


class TrainingDataset(IterableDataset):
    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int,
        seq_len: int,
        device: str | int,
        chunk_rows: int,
        shuffle: bool,
        seed: int | None,
        vocab_size: int,
        uniform_negative_items: int,
        in_batch_negative_items: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.chunk_rows = chunk_rows
        self.shuffle = shuffle
        self.seed = seed
        self.vocab_size = vocab_size
        self.uniform_negative_items = uniform_negative_items
        self.in_batch_negative_items = in_batch_negative_items
        self.min_freq = 1e-12
        self.epoch = 0
        self.ddp = False
        self.rank = 0
        self.world_size = 1
        self.usable_batches = 0

        self.batch_num_tokens = batch_size * seq_len + 1
        self.uniform_prob = float(uniform_negative_items) / max(self.vocab_size - 1, 1)

        row_lens = df.get_column("token_id").list.len().to_numpy()
        tokens = df.explode("token_id").get_column("token_id").cast(int).to_numpy()

        self.total_num_tokens = int(tokens.shape[0])
        self._all_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        self._row_lens = torch.tensor(row_lens, dtype=torch.long, device=device)
        self._row_starts = torch.cumsum(self._row_lens, 0) - self._row_lens

        self._epoch_tokens = self._all_tokens

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def _in_batch_negative_q(
        self, batch_token_ids: torch.Tensor, sampled_token_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_tokens = batch_token_ids.flatten().to(torch.long)
        batch_size = max(int(batch_tokens.numel()), 1)

        valid_batch_mask = (batch_tokens >= 0) & (batch_tokens < self.vocab_size)
        uniq, counts = torch.unique(batch_tokens[valid_batch_mask], return_counts=True)

        sampled_token_ids = sampled_token_ids.to(torch.long)
        sampled_q = torch.full(
            sampled_token_ids.shape,
            self.min_freq,
            dtype=torch.float32,
            device=sampled_token_ids.device,
        )
        valid_sampled_mask = (sampled_token_ids >= 0) & (
            sampled_token_ids < self.vocab_size
        )

        if uniq.numel() > 0:
            valid_sampled = sampled_token_ids[valid_sampled_mask]
            pos = torch.bucketize(valid_sampled, uniq).clamp(max=uniq.numel() - 1)
            matched = uniq[pos] == valid_sampled
            freq = torch.where(matched, counts[pos], torch.zeros_like(counts[pos]))
            sampled_q[valid_sampled_mask] = freq.to(torch.float32) / float(batch_size)

        sampled_q[sampled_token_ids == BOS] = 1.0
        return torch.clamp(sampled_q, min=self.min_freq)

    def ddp_setup(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.ddp = True

    def _epoch_token_stream(self) -> torch.Tensor:
        n_rows = self._row_lens.numel()
        if self.shuffle:
            if self.seed is not None:
                gen = torch.Generator(device=self.device)
                gen.manual_seed(self.seed + self.epoch)
                perm = torch.randperm(n_rows, generator=gen, device=self.device)
            else:
                perm = torch.randperm(n_rows, device=self.device)
        else:
            perm = torch.arange(n_rows, device=self.device)

        if self.ddp:
            lens = self._row_lens[perm]
            shard_tokens = min(
                int(lens[r :: self.world_size].sum())
                for r in range(self.world_size)
            )
            self.usable_batches = shard_tokens // self.batch_num_tokens
            perm = perm[self.rank :: self.world_size]
        else:
            self.usable_batches = self.total_num_tokens // self.batch_num_tokens

        lens = self._row_lens[perm]
        starts = self._row_starts[perm]

        out_starts = torch.cumsum(lens, 0) - lens
        row_of = torch.repeat_interleave(
            torch.arange(lens.numel(), device=self.device), lens
        )
        within = torch.arange(int(lens.sum()), device=self.device) - out_starts[row_of]
        gather = starts[row_of] + within
        return self._all_tokens[gather]

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._epoch_tokens = self._epoch_token_stream()

    def create_batch(self, t: torch.Tensor) -> TrainingBatch:
        inputs = t[:-1].view((self.batch_size, self.seq_len))
        targets = t[1:].view((self.batch_size, self.seq_len))
        uniform_negatives = torch.randint(
            1,
            self.vocab_size,
            (self.uniform_negative_items,),
            device=self.device,
        )

        ndx = torch.randint(
            0,
            self.batch_num_tokens - 1,
            (self.in_batch_negative_items,),
            device=self.device,
        )
        in_batch_negatives = t.flatten()[ndx]
        in_batch_mask = in_batch_negatives != BOS

        negatives = torch.concat(
            [uniform_negatives, in_batch_negatives[in_batch_mask]]
        )

        positive_q = torch.full_like(
            targets, self.uniform_prob, dtype=torch.float32
        ) + self.in_batch_negative_items * self._in_batch_negative_q(
            t, targets.flatten()
        ).view_as(targets)

        negative_q = torch.full(
            (negatives.numel(),),
            self.uniform_prob,
            dtype=torch.float32,
            device=self.device,
        ) + self.in_batch_negative_items * self._in_batch_negative_q(
            t, negatives
        )

        return TrainingBatch(
            inputs=inputs,
            targets=targets,
            negatives=negatives,
            size=inputs.numel(),
            negative_log_q=negative_q.log(),
            positive_log_q=positive_q.log(),
        )

    def __iter__(self):
        bnt = self.batch_num_tokens
        for i in range(self.usable_batches):
            chunk = self._epoch_tokens[i * bnt : i * bnt + bnt]
            yield self.create_batch(chunk)
