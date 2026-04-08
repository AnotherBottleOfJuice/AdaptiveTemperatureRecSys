from dataclasses import dataclass
import math

import torch
from torch.nn.utils.rnn import pad_sequence
import polars as pl

from config import *


def get_general_data(path_interactions: str = PATH_INTERACTIONS,
                     path_embeddings: str = PATH_EMBEDDINGS,
                     path_artists: str = PATH_ARTISTS,
                     core_min_interaction_per_user: int = CORE_MIN_INTERACTIONS_PER_ITEM,
                     test_interval_seconds: int = TEST_INTERVAL_SECONDS, ):
    interactions = pl.read_parquet(path_interactions)
    embeddings = pl.read_parquet(path_embeddings)
    artists = pl.read_parquet(path_artists)

    interactions = interactions.join(embeddings, on="item_id", how="semi")

    count = interactions.group_by('item_id').len()

    interactions = (interactions.join(count, on='item_id', how='left')
                    .filter(pl.col('len') >= core_min_interaction_per_user))

    interactions = interactions.join(artists, on='item_id', how='semi')

    end = interactions['timestamp'].max()

    train = interactions.filter(pl.col('timestamp') < end - test_interval_seconds)
    test = interactions.filter(pl.col('timestamp') >= end - test_interval_seconds)
    test = test.filter(pl.col('uid').is_in(train['uid'].implode()))

    test_targets = {
        i[0]: i[1]
        for i in test.group_by('uid').agg('item_id').rows()
    }

    embeddings = embeddings.filter(pl.col('item_id').is_in(interactions['item_id'].implode()))

    return train, test, embeddings, artists, test_targets


def get_item_to_token(train, vocab_size: int = None):
    return (
        train
        .select("item_id")  # берём только item_id (проще и быстрее)
        .group_by('item_id').len().rename({'len': 'count'})
        .sort('item_id')
        .reverse()
        .sort('count')
        .reverse()
        .slice(0, vocab_size if vocab_size else None)
        .with_row_index(name='token_id')
        .with_columns(pl.col('token_id') + 1)
        .select(('item_id', 'token_id'))
    )


@dataclass
class TestBatch:
    token_ids: torch.Tensor  # [B, T_max]
    lengths: torch.Tensor  # [B]


class TestDataset:
    def __init__(
            self,
            df: pl.DataFrame,
            batch_size: int,
            device: str = "cuda",
    ):
        self.df = df
        self.batch_size = batch_size
        self.device = device

        self._tensors = [torch.tensor(i) for i in df['token_id']]

    def __len__(self):
        return math.ceil(len(self._tensors) / self.batch_size)

    def __iter__(self):
        for i in range(len(self)):
            seqs = self._tensors[i * self.batch_size:
                                 min((i + 1) * self.batch_size, len(self._tensors))]

            batch = pad_sequence(
                seqs,
                batch_first=True
            )
            lengths = torch.tensor([len(j) for j in seqs])
            yield TestBatch(batch, lengths)


def get_test_users(test: pl.DataFrame) -> pl.DataFrame:
    return (
        test
        .select("uid")
        .unique()
    )


def get_test_users_events(test: pl.DataFrame, train_events: pl.DataFrame,
                          max_len: int = MAX_LEN) -> pl.DataFrame:
    return (
        train_events
        .join(get_test_users(test), on="uid", how="semi")
        .sort("timestamp")
        .group_by("uid", maintain_order=True)
        .tail(n=max_len - 2)
        .select(["uid", "token_id"])
    )


def get_test_histories(test: pl.DataFrame, train_events: pl.DataFrame,
                       bos: int = BOS) -> pl.DataFrame:
    test_histories = (
        pl.concat([
            get_test_users(test).with_columns(pl.lit(bos, dtype=pl.UInt32).alias('token_id')),
            get_test_users_events(test, train_events),
        ])
        .group_by('uid', maintain_order=True)
        .agg('token_id')
    )

    return (
        test_histories
        .with_columns(length=pl.col("token_id").list.len())
        .sort("length", descending=True)
    )


import polars as pl

from config import MAX_TRAIN_EVENTS_PER_USER, BOS


def get_train_events(train: pl.DataFrame, item_to_token_id: pl.DataFrame,
                     max_events_per_user: int = MAX_TRAIN_EVENTS_PER_USER) -> pl.DataFrame:
    train_events = (
        train
        .join(item_to_token_id, on='item_id', how='inner')
        .sort('timestamp')
        .drop(['is_organic', 'len', 'item_id'])
    )

    return (
        train_events
        .group_by('uid', maintain_order=True)
        .tail(max_events_per_user - 1)
    )


def get_train_histories(train_events: pl.DataFrame,
                        bos: int = BOS) -> pl.DataFrame:
    return (
        pl.concat([
            (
                train_events.select('uid').unique()
                .with_columns(pl.lit(bos, dtype=pl.UInt32).alias('token_id'))
            ),
            train_events.drop('timestamp')
        ])
        .group_by('uid', maintain_order=True)
        .agg('token_id')
    )


from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from config import VOCAB_SIZE, UNIFORM_NEGATIVES_NUM, IN_BATCH_NEGATIVES_NUM


@dataclass
class TrainingBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    negatives: torch.Tensor
    size: int
    negative_log_q: torch.Tensor | None = None
    positive_log_q: torch.Tensor | None = None


class TrainingDataset:
    def __init__(
            self,
            df: pl.DataFrame,
            batch_size: int,
            seq_len: int,
            device: str = "cuda",
            chunk_rows: int = 64_000,
            shuffle: bool = True,
            seed: int | None = 42,
            pin_memory: bool = True,
            vocab_size: int = VOCAB_SIZE,
            uniform_negative_items: int = UNIFORM_NEGATIVES_NUM,
            in_batch_negative_items: int = IN_BATCH_NEGATIVES_NUM,
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

        self.batch_num_tokens = batch_size * seq_len + 1
        self.total_num_tokens = int(self.df.get_column("token_id").list.len().sum())

        uniform_prob = 1.0 / max(self.vocab_size - 1, 1)
        inbatch_prob = 1.0 / max(self.batch_num_tokens - 1, 1)
        self.uniform_log_q = float(np.log(uniform_prob))
        self.inbatch_log_q = float(np.log(inbatch_prob))

    def __len__(self):
        return self.total_num_tokens // self.batch_num_tokens

    def __iter__(self):
        df = self.df

        if self.shuffle:
            df = df.sample(fraction=1, shuffle=True, seed=self.seed)

        buf = np.ndarray((0,), dtype=np.int64)
        for i in range(0, df.height // self.chunk_rows + 1):
            chunk = df.slice(i * self.chunk_rows,
                             min(df.height - i * self.chunk_rows, self.chunk_rows))
            tokens = chunk.explode('token_id').get_column('token_id').cast(int).to_numpy()
            buf = np.concat((buf, tokens))
            while len(buf) >= self.batch_num_tokens + 1:
                if torch.cpu.is_available():
                    t_cpu = torch.as_tensor(buf[:self.batch_num_tokens], device='cpu')
                else:
                    t_cpu = torch.as_tensor(buf[:self.batch_num_tokens], device='mps')

                if self.pin_memory:
                    t_cpu = t_cpu.pin_memory()
                t = t_cpu.to(self.device)
                inputs = t[:-1].view((self.batch_size, self.seq_len))
                targets = t[1:].view((self.batch_size, self.seq_len))
                uniform_negatives = torch.randint(1, self.vocab_size, (self.uniform_negative_items,),
                                                  device=self.device)

                ndx = torch.randint(0, self.batch_num_tokens - 1, (self.in_batch_negative_items,), device=self.device)
                in_batch_negatives = t.flatten()[ndx]

                negatives = torch.concat([uniform_negatives, in_batch_negatives])
                negative_log_q = torch.concat([
                    torch.full((uniform_negatives.numel(),), self.uniform_log_q, device=self.device),
                    torch.full((in_batch_negatives.numel(),), self.inbatch_log_q, device=self.device),
                ])
                positive_log_q = torch.full_like(targets, self.inbatch_log_q)

                yield TrainingBatch(inputs=inputs, targets=targets,
                                    negatives=negatives,
                                    size=inputs.numel(),
                                    negative_log_q=negative_log_q,
                                    positive_log_q=positive_log_q)
                buf = buf[self.batch_num_tokens:]
