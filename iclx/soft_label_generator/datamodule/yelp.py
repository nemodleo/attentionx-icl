from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class YelpDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("yelp_review_full")
        self.train_dataset = self._tokenize(dataset['train'])
        self.val_dataset = self._tokenize(dataset['test'])
        self.test_dataset = self._tokenize(dataset['test'])

    def label_texts(self):
        return ["1 star", "2 star", "3 star", "4 star", "5 star"]
