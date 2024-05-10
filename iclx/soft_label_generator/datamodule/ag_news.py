from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class AGNewsDataModule(BaseDataModule):
    def setup(self, stage='all'):
        dataset = load_dataset("ICKD/agnews")

        if stage == 'train' or stage == 'all':
            train_dataset = dataset['train']
            if self.sampling_rate < 1.0:
                train_dataset = train_dataset.filter(
                    lambda example, idx: idx % 10 < self.sampling_rate * 10,
                    with_indices=True,
                )
            self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'valid' or stage == 'all':
            val_dataset = dataset['valid']
            self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'test' or stage == 'all':
            test_dataset = dataset['test']
            self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def label_texts(self):
        return ["World", "Sports", "Business", "Sci/Tech"]