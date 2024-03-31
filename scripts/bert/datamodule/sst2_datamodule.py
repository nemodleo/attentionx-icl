from datasets import load_dataset

from torch.utils.data import DataLoader

import pytorch_lightning as pl

from transformers import AutoTokenizer


class SST2DataModule(pl.LightningDataModule):
    def __init__(self, model_name_or_path: str, batch_size: int):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.batch_size = batch_size

    def setup(self, stage=None):
        dataset = load_dataset("glue", "sst2")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['validation'])
        self.test_dataset = self._tokenize(dataset['test'])

    def _tokenize(self, dataset):
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples['sentence'],
                padding=True,
                truncation=True,
                max_length=512,
            ),
            batched=True,
        )
        return tokenized_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def num_labels(self):
        return 2
