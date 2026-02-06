from typing import Literal

from torch.utils.data import DataLoader
import torch

from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from pandas import DataFrame
import os

from functools import partial

from config import NUM_WORKERS_FOR_LOADER

class YambdaDataset(Dataset):

    DEFAULT_PATH = './datasets/yambda_likes_dataset'
    SECONDS_IN_DAY = 24 * 60 * 60

    def __init__(self, path : str | None = None,
                 dataset_type : Literal['50m', '500m', '5b'] = '50m',
                 overwrite : bool = False,
                 mode : Literal['train', 'val'] = 'train',
                 max_seq_len : int = 256):

        self.path = path if path is not None else YambdaDataset.DEFAULT_PATH
        self.path += dataset_type

        self.dataset = None

        if overwrite or not os.path.exists(self.path):
            self.dataset = load_dataset("yandex/yambda", data_dir=f"flat/{dataset_type}", data_files="likes.parquet")
            self.dataset.save_to_disk(self.path)
        else:
            self.dataset = load_from_disk(self.path)

        self.dataset = DataFrame(self.dataset['train'])
        self.dataset : DataFrame

        self.pad_id = self.dataset['item_id'].max() + 1

        start = self.dataset['timestamp'].min()
        end = self.dataset['timestamp'].max()

        if mode == 'train':
            self.dataset = self.dataset[self.dataset['timestamp'] < end - 7 * self.SECONDS_IN_DAY]
        else:
            self.dataset = self.dataset[self.dataset['timestamp'] >= end - 7 * self.SECONDS_IN_DAY]

        self.dataset : DataFrame
        self.dataset.sort_values(by=['timestamp'], inplace=True)
        self.dataset['num'] = self.dataset.groupby('uid').cumcount() // max_seq_len
        self.dataset = self.dataset.groupby(['uid', 'num'])['item_id'].apply(list).reset_index()

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.tensor(self.dataset.iloc[idx]['item_id'], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.dataset)


class Utils:

    @staticmethod
    def collate_to_batch(batch, pad_id, max_len):
        batch_t = torch.tensor([[seq[i] if i < len(seq) else pad_id for i in range(max_len)] for seq in batch], dtype=torch.long)
        return batch_t

    @staticmethod
    def collate_with_random_negatives(batch, pad_id, num_neg, max_len):
        batch_t = Utils.collate_to_batch(batch, pad_id, max_len)
        neg = torch.randint(0, pad_id, (*batch_t.shape, num_neg), dtype=torch.long)
        return [batch_t, neg]

    @staticmethod
    def get_train_dataloader(batch_size=32, max_len=200, num_neg=256,
                            dataset_type: Literal['50m', '500m', '5b'] = '500m'):
        dataset = YambdaDataset(max_seq_len=max_len, mode='train', dataset_type=dataset_type)

        collate_fn = partial(
            Utils.collate_with_random_negatives,
            pad_id=dataset.pad_id,
            num_neg=num_neg,
            max_len=max_len
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True,
                                num_workers=NUM_WORKERS_FOR_LOADER,
                                collate_fn=collate_fn)
        return dataloader

    @staticmethod
    def get_val_dataloader(batch_size=32, max_len=200,
                        dataset_type: Literal['50m', '500m', '5b'] = '500m'):
        dataset = YambdaDataset(max_seq_len=max_len, mode='val', dataset_type=dataset_type)

        collate_fn = partial(
            Utils.collate_to_batch,
            batch_size=batch_size,
            max_len=max_len
        )

        dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                                num_workers=NUM_WORKERS_FOR_LOADER,
                                collate_fn=collate_fn)
        return dataloader


__all__ = ["YambdaDataset", "Utils"]