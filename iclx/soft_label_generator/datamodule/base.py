from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from transformers import AutoTokenizer


class BaseDataModule(pl.LightningDataModule, ABC):
    def __init__(self, model_name_or_path: str, batch_size: int, max_token_len: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.num_workers = 0

    @abstractmethod
    def setup(self, stage=None):
        raise NotImplementedError

    def _tokenize(self, dataset):
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['text'],
                padding=True,
                truncation=True,
                max_length=self.max_token_len,
            ),
            batched=True,
        )
        return tokenized_dataset
    
    def _collate_fn(self, batch):
        text = [x['text'] for x in batch]
        input_ids = pad_sequence(
            [torch.tensor(x['input_ids']) for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(x['attention_mask']) for x in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.tensor([x['label'] for x in batch])
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
        raise len(self.label_texts())
