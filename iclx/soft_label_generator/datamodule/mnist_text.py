from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class MNISTTextDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("nemodleo/mnist-text-small14")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['test'])
        self.test_dataset = self._tokenize(dataset['test'])

    def label_texts(self):
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
