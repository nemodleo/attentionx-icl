from datasets import load_dataset

import torch
from torch.nn.utils.rnn import pad_sequence

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class TRECDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("trec")

        train_dataset = dataset['train']
        val_dataset = dataset['test']
        test_dataset = dataset['test']

        if self.sampling_rate < 1.0:
            train_dataset = train_dataset.filter(
                lambda example, idx: idx % 10 < self.sampling_rate * 10,
                with_indices=True,
            )

        self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)
        self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)
        self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def label_texts(self):
        return ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
