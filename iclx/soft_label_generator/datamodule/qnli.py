import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class QNLIDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("SetFit/qnli")

        train_dataset = dataset['train'].map(self._merge_premise_hypothesis)
        val_dataset = dataset['validation'].map(self._merge_premise_hypothesis)
        test_dataset = dataset['test'].map(self._merge_premise_hypothesis)

        if self.sampling_rate < 1.0:
            train_dataset = train_dataset.filter(
                lambda example, idx: idx % 10 < self.sampling_rate * 10,
                with_indices=True,
            )

        self.train_dataset = self._tokenize(train_dataset)
        self.val_dataset = self._tokenize(val_dataset)
        self.test_dataset = self._tokenize(test_dataset)

    def _merge_premise_hypothesis(self, examples):
        return {
            'text': examples['text1'] + ' [SEP] ' + examples['text2'],
            'label': examples['label'],
        }

    def label_texts(self):
        return ["entailment", "not_entailment"]
    