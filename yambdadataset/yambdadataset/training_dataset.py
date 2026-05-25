from dataclasses import dataclass

import numpy as np
import polars as pl
import torch

BOS = 0


@dataclass
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor
    size: int
    negative_log_q: torch.Tensor | None
    positive_log_q: torch.Tensor | None


class TrainingDataset:
    def __init__(
        self,
        df: pl.DataFrame,
        batch_size: int,
        seq_len: int,
        device: str | int,
        chunk_rows: int,
        shuffle: bool,
        seed: int | None,
        pin_memory: bool,
        vocab_size: int,
        uniform_negative_items: int,
        in_batch_negative_items: int,
    ):
        self.df = df
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.chunk_rows = chunk_rows
        self.shuffle = shuffle
        self.seed = seed
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size
        self.uniform_negative_items = uniform_negative_items
        self.in_batch_negative_items = in_batch_negative_items
        self.min_freq = 1e-12
        self.epoch = 0
        self.ddp = False

        self.batch_num_tokens = batch_size * seq_len + 1
        self.total_num_tokens = int(self.df.get_column("token_id").list.len().sum())

        self.uniform_prob = float(uniform_negative_items) / max(self.vocab_size - 1, 1)

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def _in_batch_negative_q(
        self, batch_token_ids: torch.Tensor, sampled_token_ids: torch.Tensor
    ) -> torch.Tensor:
        batch_tokens = batch_token_ids.flatten().to(torch.long)
        batch_size = max(int(batch_tokens.numel()), 1)

        # Guard against out-of-range token ids to avoid CUDA device-side asserts.
        valid_batch_mask = (batch_tokens >= 0) & (batch_tokens < self.vocab_size)
        safe_batch_tokens = batch_tokens[valid_batch_mask]
        multiplicity = torch.bincount(safe_batch_tokens, minlength=self.vocab_size).to(
            self.device
        )
        q = multiplicity.to(dtype=torch.float32) / float(batch_size)
        q[BOS] = 1

        sampled_token_ids = sampled_token_ids.to(torch.long)
        sampled_q = torch.full(
            sampled_token_ids.shape,
            self.min_freq,
            device=self.device,
            dtype=torch.float32,
        )
        valid_sampled_mask = (sampled_token_ids >= 0) & (
            sampled_token_ids < self.vocab_size
        )
        sampled_q[valid_sampled_mask] = q[sampled_token_ids[valid_sampled_mask]]
        return torch.clamp(sampled_q, min=self.min_freq)

    def ddp_setup(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.ddp = True

    def __iter__(self):
        df = self.df

        if self.shuffle:
            seed = self.seed + self.epoch if self.seed is not None else None
            df = df.sample(fraction=1, shuffle=True, seed=seed)

        buf = np.ndarray((0,), dtype=np.int64)
        batch_counter = 0

        usable_batches = self.total_num_tokens // self.batch_num_tokens

        if self.ddp:
            usable_batches -= usable_batches % self.world_size

        for i in range(0, df.height // self.chunk_rows + 1):
            chunk = df.slice(
                i * self.chunk_rows,
                min(df.height - i * self.chunk_rows, self.chunk_rows),
            )
            tokens = (
                chunk.explode("token_id").get_column("token_id").cast(int).to_numpy()
            )
            buf = np.concat((buf, tokens))

            if batch_counter >= usable_batches:
                break

            while len(buf) >= self.batch_num_tokens:
                if not self.ddp or batch_counter % self.world_size == self.rank:
                    if torch.cpu.is_available():
                        t_cpu = torch.as_tensor(
                            buf[: self.batch_num_tokens], device="cpu"
                        )
                    else:
                        t_cpu = torch.as_tensor(
                            buf[: self.batch_num_tokens], device="mps"
                        )

                    if self.pin_memory:
                        t_cpu = t_cpu.pin_memory()

                    t = t_cpu.to(self.device)
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
                        device=self.device,
                        dtype=torch.float32,
                    ) + self.in_batch_negative_items * self._in_batch_negative_q(
                        t, negatives
                    )

                    yield TrainingBatch(
                        inputs=inputs,
                        targets=targets,
                        negatives=negatives,
                        size=inputs.numel(),
                        negative_log_q=negative_q.log(),
                        positive_log_q=positive_q.log(),
                    )

                batch_counter += 1
                buf = buf[self.batch_num_tokens :]

                if batch_counter >= usable_batches:
                    break

        self.epoch += 1
