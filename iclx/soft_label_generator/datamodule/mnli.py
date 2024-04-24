import torch

from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class MNLIDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("nyu-mll/multi_nli")

        train_dataset = dataset['train'].map(self._merge_premise_hypothesis)
        val_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)
        test_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)

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
            'text': examples['premise'] + ' [SEP] ' + examples['hypothesis'],
            'label': examples['label'],
        }

    def label_texts(self):
        return ["entailment", "neutral", "contradiction"]
    