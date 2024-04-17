import torch
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset

from iclx.soft_label_generator.datamodule.base import BaseDataModule


class QNLIDataModule(BaseDataModule):
    def setup(self, stage=None):
        dataset = load_dataset("SetFit/qnli")

        train_dataset = dataset['train'].map(self._merge_premise_hypothesis)
        val_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)
        test_dataset = dataset['validation_matched'].map(self._merge_premise_hypothesis)

        self.train_dataset = self._tokenize(train_dataset)
        self.val_dataset = self._tokenize(val_dataset)
        self.test_dataset = self._tokenize(test_dataset)

    def _merge_premise_hypothesis(self, examples):
        return {
            'text': examples['text1'] + ' [SEP] ' + examples['text2'],
            'label': examples['label'],
        }

    def _collate_fn(self, batch):
        text = [x['text'] for x in batch]
        input_ids = pad_sequence(
            [torch.tensor(x['input_ids']) for x in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(x['attention_mask']) for x in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.tensor([x['label'] for x in batch])
        return {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def label_texts(self):
        return ["entailment", "not_entailment"]
    