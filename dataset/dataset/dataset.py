import os
from functools import partial
from typing import Literal

import polars as pl
import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset


class YambdaDataset(Dataset):
    DEFAULT_PATH = './datasets/yambda_likes_dataset'
    SECONDS_IN_DAY = 24 * 60 * 60

    def __init__(self,
                 path: str | None = None,
                 dataset_type: Literal['50m', '500m', '5b'] = '50m',
                 overwrite: bool = False,
                 mode: Literal['train', 'val'] = 'train',
                 max_seq_len: int = 256):

        self.path = path if path is not None else YambdaDataset.DEFAULT_PATH
        self.path += dataset_type

        self.dataset = None

        if overwrite or not os.path.exists(self.path):
            self.dataset = load_dataset("yandex/yambda", data_dir=f"flat/{dataset_type}", data_files="likes.parquet")
            self.dataset.save_to_disk(self.path)
        else:
            self.dataset = load_from_disk(self.path)

        self.dataset = pl.from_arrow(self.dataset['train'].data.table)

        self.dataset = self.dataset.with_columns(pl.col('item_id').rank("dense"))

        self.pad_id = 0
        self.num_tokens = self.dataset.max()['item_id'].item() + 1

        sep = self.dataset.max()['timestamp'] - self.SECONDS_IN_DAY * 7

        if mode == 'train':
            self.dataset = self.dataset.filter(pl.col('timestamp') <= sep)
        else:
            self.dataset = self.dataset.filter(pl.col('timestamp') > sep)

        self.dataset = self.dataset.sort('timestamp')
        self.dataset = (
            self
            .dataset.group_by('uid', maintain_order=True)
            .agg(pl.col('item_id').tail(max_seq_len))
        )
        self.dataset: pl.DataFrame
        self.dataset = [torch.LongTensor(i) for i in self.dataset['item_id'].to_list()]

    def __getitem__(self, idx) -> torch.Tensor:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)


class Utils:

    @staticmethod
    def collate_to_batch(batch, pad_id, max_len):
        batch_t = torch.tensor([[seq[i] if i < len(seq) else pad_id for i in range(max_len)] for seq in batch],
                               dtype=torch.long)
        return batch_t

    @staticmethod
    def collate_with_random_negatives(batch, pad_id, max_token, num_neg, max_len):
        batch_t = Utils.collate_to_batch(batch, pad_id, max_len)
        neg = torch.randint(0, max_token, (*batch_t.shape, num_neg), dtype=torch.long)
        return [batch_t, neg]

    @staticmethod
    def get_train_dataloader(batch_size=32, max_len=200, num_neg=256,
                             dataset_type: Literal['50m', '500m', '5b'] = '500m',
                             num_workers=4):
        dataset = YambdaDataset(max_seq_len=max_len, mode='train', dataset_type=dataset_type)

        collate_fn = partial(
            Utils.collate_with_random_negatives,
            pad_id=dataset.pad_id,
            max_token=dataset.num_tokens,
            num_neg=num_neg,
            max_len=max_len
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers,
                                collate_fn=collate_fn)
        return dataloader

    @staticmethod
    def get_val_dataloader(batch_size=32, max_len=200,
                           dataset_type: Literal['50m', '500m', '5b'] = '500m',
                           num_workers=4):
        dataset = YambdaDataset(max_seq_len=max_len, mode='val', dataset_type=dataset_type)

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
