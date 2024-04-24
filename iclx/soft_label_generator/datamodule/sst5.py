from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class SST5DataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("SetFit/sst5")

        train_dataset = dataset['train']

        if self.sampling_rate < 1.0:
            train_dataset = train_dataset.filter(
                lambda example, idx: idx % 10 < self.sampling_rate * 10,
                with_indices=True,
            )

        self.train_dataset = self._tokenize(train_dataset)
        self.val_dataset = self._tokenize(dataset['validation'])
        self.test_dataset = self._tokenize(dataset['test'])

    def label_texts(self):
        return ["very negative", "negative", "neutral", "positive", "very positive"]
    