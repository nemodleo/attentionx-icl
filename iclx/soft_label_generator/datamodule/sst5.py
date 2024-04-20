from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class SST5DataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("SetFit/sst5")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['validation'])
        self.test_dataset = self._tokenize(dataset['test'])

    def label_texts(self):
        return ["very negative", "negative", "neutral", "positive", "very positive"]
    