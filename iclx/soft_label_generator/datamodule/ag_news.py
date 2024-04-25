from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class AGNewsDataModule(BaseDataModule):
    def setup(self, stage='all'):
        dataset = load_dataset("ag_news")

        if stage == 'train' or stage is 'all':
            train_dataset = dataset['train']
            if self.sampling_rate < 1.0:
                train_dataset = train_dataset.filter(
                    lambda example, idx: idx % 10 < self.sampling_rate * 10,
                    with_indices=True,
                )
            self.train_dataset = self._tokenize(train_dataset)

        if stage == 'validation' or stage is 'all':
            val_dataset = dataset['test']
            self.val_dataset = self._tokenize(val_dataset)

        if stage == 'test' or stage == 'all':
            test_dataset = dataset['test']
            self.test_dataset = self._tokenize(test_dataset)

    def label_texts(self):
        return ["World", "Sports", "Business", "Sci/Tech"]

