from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from transformers import AutoTokenizer


class BaseDataSet(Dataset):
    def __init__(self, dataset, model_name_or_path, max_token_len):
        self.super().__init__()
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_token_len = max_token_len

    def __getitem__(self, idx):
        example = self.dataset[idx]
        result = self.tokenizer(
            example['text'],
            padding="max_length",
            truncation=True,
            max_length=self.max_token_len,
        )
        return {
            'text': result['text'],
            'input_ids': torch.tensor(result['input_ids']),
            'attention_mask': torch.tensor(result['attention_mask']),
            'labels': torch.tensor(result['label'])
        }

    def __len__(self):
        return len(self.dataset)



class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, model_name_or_path: str, batch_size: int, max_token_len: int, sampling_rate: float, num_workers: int = 16):
        super().__init__()
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.sampling_rate = sampling_rate
        self.num_workers = num_workers

    @abstractmethod
    def setup(self, stage=None):
        raise NotImplementedError
    

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    @abstractmethod
    def label_texts(self):
        raise NotImplementedError

    def num_labels(self):
        return len(self.label_texts())
