import fire
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification

from iclx.soft_label_generator.datamodule.sst2 import SST2DataModule
from iclx.soft_label_generator.datamodule.sst5 import SST5DataModule
from iclx.soft_label_generator.datamodule.trec import TRECDataModule
from iclx.soft_label_generator.datamodule.ag_news import AGNewsDataModule


def infer(
    model_name_or_path: str = "bert-base-uncased",
    dataset_name: str = "sst2",
    dataset_split: str = "train",
    batch_size: int = 512,
    output_path: str = "result.csv"
):
    # Load data
    if dataset_name == "sst2":
        data_module = SST2DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
        )
    elif dataset_name == "sst5":
        data_module = SST5DataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
        )
    elif dataset_name == "trec":
        data_module = TRECDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
        )
    elif dataset_name == "ag_news":
        data_module = AGNewsDataModule(
            model_name_or_path=model_name_or_path,
            batch_size=batch_size,
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
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
    model.eval()

    # Start inference
    data = {
        "index": [],
        "sentence": [],
        "label": [],
        "positive_prob": [],
        "negative_prob": []
    }

    for batch in tqdm(dataloader):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=-1)
            positive_prob = probs[:, 1]
            negative_prob = probs[:, 0]

        idx = len(data["index"])
        label = "positive" if labels[0].item() else "negative"

        data["index"].append(idx)
        data["label"].append(label)
        data["positive_prob"].append(positive_prob.tolist())
        data["negative_prob"].append(negative_prob.tolist())

    pd.DataFrame(data).to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(infer)
