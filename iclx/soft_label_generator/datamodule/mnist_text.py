from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class MNISTTextDataModule(BaseDataModule):
    def setup(self, stage='all'):
        dataset = load_dataset("ICKD/mnist-text-small")
        if stage == 'train' or stage == 'all':
            train_dataset = dataset['train']
            self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'validation' or stage == 'all':
            val_dataset = dataset['test']
            self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'test' or stage == 'all':
            test_dataset = dataset['test']
            self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def label_texts(self):
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
