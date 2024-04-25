from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class SST5DataModule(BaseDataModule):
    def setup(self, stage='all'):
        dataset = load_dataset("SetFit/sst5")

        if stage == 'train' or stage is 'all':
            train_dataset = dataset['train']
            if self.sampling_rate < 1.0:
                train_dataset = train_dataset.filter(
                    lambda example, idx: idx % 10 < self.sampling_rate * 10,
                    with_indices=True,
                )
             self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)
        
        if stage == 'validation' or stage is 'all':
            val_dataset = dataset['validation']
            self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)

        if stage == 'test' or stage == 'all':
            test_dataset = dataset['test']
            self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def label_texts(self):
        return ["very negative", "negative", "neutral", "positive", "very positive"]
