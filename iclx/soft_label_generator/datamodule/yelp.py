from datasets import load_dataset

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from transformers import AutoTokenizer


class YelpDataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path: str, batch_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = load_dataset("yelp_review_full")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['test'])
        self.test_dataset = self._tokenize(dataset['test'])

    def _tokenize(self, dataset):
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['text'],
                padding=True,
                truncation=True,
                max_length=512,
            ),
            batched=True,
        )
        return tokenized_dataset
    
    def _collate_fn(self, batch):
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
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )

    def num_labels(self):
        return 5
