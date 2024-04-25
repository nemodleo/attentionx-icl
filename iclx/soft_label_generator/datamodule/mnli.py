import torch

from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class MNLIDataModule(BaseDataModule):
    def setup(self, stage='all'):
        dataset = load_dataset("nyu-mll/multi_nli")

        if stage == 'train' or stage == 'all':
            train_dataset = dataset['train'].map(self._merge_premise_hypothesis)
            if self.sampling_rate < 1.0:
                train_dataset = train_dataset.filter(
                    lambda example, idx: idx % 10 < self.sampling_rate * 10,
                    with_indices=True,
                )
            self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'validation' or stage == 'all':
            val_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)
            self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'test' or stage == 'all':
            test_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)
            self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def _merge_premise_hypothesis(self, examples):
        return {
            'text': examples['premise'] + ' [SEP] ' + examples['hypothesis'],
            'label': examples['label'],
        }

    def label_texts(self):
        return ["entailment", "neutral", "contradiction"]
