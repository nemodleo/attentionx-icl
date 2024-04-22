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
        labels = torch.tensor([x['coarse_label'] for x in batch])
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
    def label_texts(self):
        return ["DESC", "ENTY", "ABBR", "HUM", "NUM", "LOC"]
