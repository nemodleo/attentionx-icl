from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class AGNewsDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("ag_news")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['test'])
        self.test_dataset = self._tokenize(dataset['test'])
    
    def label_texts(self):
        return ["World", "Sports", "Business", "Sci/Tech"]
