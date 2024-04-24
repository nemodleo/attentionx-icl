from datasets import load_dataset

import torch
from torch.nn.utils.rnn import pad_sequence

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class TRECDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("trec")

        train_dataset = dataset['train']

        if self.sampling_rate < 1.0:
            train_dataset = train_dataset.filter(
                lambda example, idx: idx % 10 < self.sampling_rate * 10,
                with_indices=True,
            )

        self.train_dataset = self._tokenize(train_dataset)
        self.val_dataset = self._tokenize(dataset['test'])
        self.test_dataset = self._tokenize(dataset['test'])

    def label_texts(self):
        return ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
