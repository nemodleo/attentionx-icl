import os
import json
from typing import List
from typing import Dict
from typing import Any

import fire
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification

from iclx.soft_label_generator.datamodule.sst2 import SST2DataModule
from iclx.soft_label_generator.datamodule.sst5 import SST5DataModule
from iclx.soft_label_generator.datamodule.trec import TRECDataModule
from iclx.soft_label_generator.datamodule.ag_news import AGNewsDataModule
from iclx.soft_label_generator.datamodule.yelp import YelpDataModule
from iclx.soft_label_generator.datamodule.mnli import MNLIDataModule
from iclx.soft_label_generator.datamodule.qnli import QNLIDataModule


def save_to_jsonl(data: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')   


def infer(
    checkpoint_path: str,
    model_name_or_path: str = "bert-base-uncased",
    dataset_name: str = "sst2",
    dataset_split: str = "train",
    batch_size: int = 512,
    max_token_len: int = 512,
    file_name: str = "result.jsonl"
):
    # Load data
    if dataset_name == "sst2":
        data_module = SST2DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "sst5":
        data_module = SST5DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "trec":
        data_module = TRECDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "ag_news":
        data_module = AGNewsDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "yelp":
        data_module = YelpDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "mnli":
        data_module = MNLIDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    elif dataset_name == "qnli":
        data_module = QNLIDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
            max_token_len=max_token_len,
        )
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    
    data_module.setup()

    if dataset_split == "train":
        dataloader = data_module.train_dataloader()
    elif dataset_split == "val":
        dataloader = data_module.val_dataloader()
    elif dataset_split == "test":
        dataloader = data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid dataset split: {dataset_split}")

    # Load model
    num_labels = data_module.num_labels()
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
    model.eval()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if 'model.' in k}
    model.load_state_dict(checkpoint['state_dict'])

    # Start inference
    data = []

    for batch in tqdm(dataloader):
        texts = batch['text']
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            pseudo_gt = F.softmax(outputs.logits, dim=-1).max(dim=-1)[1].cpu().numpy().tolist()
            pseudo_gt_text = [data_module.label_texts()[label] for label in pseudo_gt]
            probs = probs.cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            label_texts = [data_module.label_texts()[label] for label in labels]
            data.extend([
                {
                    "text": text,
                    "label": label,
                    "label_text": label_text,
                    "pseudo_gt": pseudo_gt,
                    "pseudo_gt_text": pseudo_gt_text,
                    **{str(i): prob for i, prob in enumerate(probs_)}
                }
                for text, label, label_text, probs_ in zip(texts, labels, label_texts, probs)
            ])

    output_path = f"./data/{dataset_name}/{file_name}"
    save_to_jsonl(data, output_path)
        

if __name__ == "__main__":
    fire.Fire(infer)
