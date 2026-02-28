import os
from functools import partial
from typing import Literal

import datasets
import polars as pl
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class YambdaDataset(Dataset):
    DEFAULT_PATH = './datasets/yambda_likes_dataset'
    SECONDS_IN_DAY = 24 * 60 * 60

    def __init__(self,
                 path: str | None = None,
                 dataset_type: Literal['50m', '500m', '5b'] = '50m',
                 overwrite: bool = False,
                 mode: Literal['train', 'val', 'eval'] = 'train',
                 seq_len: int = 256):

        if mode == 'eval':
            raise NotImplementedError("Eval mode is not implemented yet.")

        self.path = path if path is not None else YambdaDataset.DEFAULT_PATH
        self.path += dataset_type

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.dataset = None

        if overwrite or not os.path.exists(self.path):
            self.dataset = load_dataset("yandex/yambda", data_dir=f"flat/{dataset_type}", data_files="likes.parquet")
            self.dataset: datasets.Dataset
            self.dataset = self.dataset['train'].to_polars()
            self.dataset.write_parquet(self.path)
        else:
            self.dataset = pl.read_parquet(self.path)
        self.dataset: pl.DataFrame
        self.dataset = self.dataset.with_columns(pl.col('item_id') + 1)

        self.num_tokens = self.dataset.unique('item_id').count().item(0, 0) + 1
        self.BOS = 0
        self.pad_id = 0
        self.UNK = 0
        self.mode = mode
        self.seq_len = seq_len
        self.popular_token = None

    def cast_to_mode(self):
        self.num_tokens = self.dataset.max()['item_id'].item() + 1
        sep = self.dataset.max()['timestamp'] - self.SECONDS_IN_DAY * 7

        if self.mode == 'train':
            self.dataset = self.dataset.filter(pl.col('timestamp') <= sep)
        elif self.mode == 'val':
            self.dataset = self.dataset.filter(pl.col('timestamp') > sep)

    def collate_to_seq(self):
        self.dataset = self.dataset.sort('timestamp')
        self.dataset = (
            self.dataset
            .group_by('uid', maintain_order=True)
            .agg(
                pl.col('item_id')
                .tail(self.seq_len - 1)
                .reverse()
                .append(self.BOS)
                .reverse()
                .alias('seqs'))
        )

        if self.mode == 'train':
            self.dataset = (self.dataset
                            .select(pl.col('seqs'))
                            .explode('seqs')
                            .select(pl.col('seqs').implode())
                            .item(0, 0))
            self.dataset = [torch.LongTensor(self.dataset[i * self.seq_len:
                                                          (i + 1) * self.seq_len].to_list())
                            for i in range(len(self.dataset) // self.seq_len)]
        else:
            self.dataset = [torch.LongTensor(i)
                            for i in self.dataset['seqs'].to_list()]

    def get_topk(self, k):
        if self.popular_token is None or k != len(self.popular_token):
            self.popular_token = (self.dataset.group_by('item_id')
                                  .agg(pl.len().alias('count'))
                                  .sort(pl.col('count')).select('item_id')
                                  .tail(k).to_series())
        return self.popular_token

    def filter_by_topk(self, popular_tokens: pl.Series):

        mapping = popular_tokens.to_frame().with_row_index('new_id', offset=1)

        self.num_tokens = len(popular_tokens) + 1
        self.UNK = self.num_tokens

        if self.mode == 'train':
            self.dataset = (self.dataset.sort('timestamp').join(mapping, on='item_id', how='left'))
            print(self.dataset.select('new_id').null_count() / self.dataset.select('item_id').count())
            self.dataset = (self.dataset
                            .with_columns(pl.col('new_id').alias('item_id'))
                            .with_columns(pl.col('item_id').fill_null(self.BOS)))
        else:
            self.dataset = (self.dataset.join(mapping, on='item_id', how='left'))
            print(self.dataset.select(pl.col('new_id').null_count()) / self.dataset.select(pl.col('item_id')).count())
            self.dataset = (self.dataset.with_columns(pl.col('new_id').fill_null(self.UNK))
                            .with_columns(pl.col('new_id').alias('item_id')))

    def __getitem__(self, idx) -> torch.Tensor:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class Utils:

    @staticmethod
    def collate_to_batch(batch, pad_id, max_len):
        batch_t = pad_sequence(batch, batch_first=True, padding_value=pad_id)
        batch_t = batch_t.long()
        if batch_t.shape[1] < max_len:
            shape = list(batch_t.shape)
            shape[1] = max_len - shape[1]
            batch_t = torch.concat((batch_t,
                                    torch.ones(*shape, dtype=torch.long) * pad_id), 1)
        return batch_t

    @staticmethod
    def collate_with_random_negatives(batch, max_token, num_neg):
        neg = torch.randint(0, max_token, (num_neg,), dtype=torch.long)
        return [torch.stack(batch), neg]

    @staticmethod
    def get_train_dataloader(dataset: YambdaDataset, batch_size=32, num_neg=256, num_workers=2):
        collate_fn = partial(
            Utils.collate_with_random_negatives,
            max_token=dataset.num_tokens,
            num_neg=num_neg,
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers,
                                collate_fn=collate_fn)
        return dataloader

    @staticmethod
    def get_val_dataloader(dataset: YambdaDataset, batch_size=32, max_len=256, num_workers=2):
        collate_fn = partial(
            Utils.collate_to_batch,
            pad_id=dataset.pad_id,
            max_len=max_len
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                num_workers=num_workers,
                                collate_fn=collate_fn)
        return dataloader


__all__ = ["YambdaDataset", "Utils"]
