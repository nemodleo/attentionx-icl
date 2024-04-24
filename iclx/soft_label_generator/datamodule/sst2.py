from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule, BaseDataSet


class SST2DataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("gpt3mix/sst2")

        train_dataset = dataset['train']
        val_dataset = dataset['validation']
        test_dataset = dataset['test']

        if self.sampling_rate < 1.0:
            train_dataset = train_dataset.filter(
                lambda example, idx: idx % 10 < self.sampling_rate * 10,
                with_indices=True,
            )

        self.train_dataset = BaseDataSet(train_dataset, self.model_name_or_path, self.max_token_len)
        self.val_dataset = BaseDataSet(val_dataset, self.model_name_or_path, self.max_token_len)
        self.test_dataset = BaseDataSet(test_dataset, self.model_name_or_path, self.max_token_len)

    def label_texts(self):
        return ["positive", "negative"]
