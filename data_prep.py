import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from transformers import AutoTokenizer


class ReadabilityDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_len: int = 248,
        target_sample: bool = False
    ) -> None:
        self.data = data.reset_index()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.target_sample = target_sample

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        row = self.data.iloc[index]
        text = row.excerpt
        text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]

        target = row.target
        std_error = row.standard_error

        if self.target_sample:
            print(target)
            target = self.add_target_sample(target, std_error)
            print("after:", target)

        target = torch.tensor(target, dtype=torch.float)

        return input_ids, attention_mask, target, row.id

    def add_target_sample(self, target, std):
        return np.random.normal(target, std)


def make_loaders(
    data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    fold: int,
    train_bs: int = 16,
    valid_bs: int = 32,
    num_workers: int = 4,
    target_sample: bool = False
) -> Tuple[DataLoader, DataLoader]:
    train = data[data['kfold'] != fold].reset_index(drop=True)
    valid = data[data['kfold'] == fold].reset_index(drop=True)

    train_dataset = ReadabilityDataset(train, tokenizer, target_sample=target_sample)
    valid_dataset = ReadabilityDataset(valid, tokenizer)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, valid_loader
