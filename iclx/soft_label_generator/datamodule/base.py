from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, model_name_or_path: str, batch_size: int, max_token_len: int, sampling_rate: float):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.sampling_rate = sampling_rate
        self.num_workers = 8

    @abstractmethod
    def setup(self, stage=None):
        raise NotImplementedError

    def _tokenize(self, dataset):
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.max_token_len,
            ),
            batched=True,
        )
        return tokenized_dataset
    
    def _collate_fn(self, batch):
        text = [x['text'] for x in batch]
        input_ids = torch.stack([torch.tensor(x['input_ids']) for x in batch])
        attention_mask = torch.stack([torch.tensor(x['attention_mask']) for x in batch])
        labels = torch.stack([torch.tensor(x['label']) for x in batch])
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
        )
    
    @abstractmethod
    def label_texts(self):
        raise NotImplementedError

    def num_labels(self):
        return len(self.label_texts())
