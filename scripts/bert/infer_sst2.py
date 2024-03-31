import fire
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SST2Dataset(Dataset):
    def __init__(self, tokenized_texts, labels):
        self.tokenized_texts = tokenized_texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_texts.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item


def infer(
    model_name_or_path: str,
    dataset_name: str = "sst2",
    dataset_split: str = "train",
    batch_size: int = 512,
    output_path: str = "result.csv"
):
    # Load data
    dataset = load_dataset(dataset_name, split=dataset_split)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenized_texts = tokenizer(dataset["sentence"], padding=True, truncation=True, return_tensors="pt")
    labels = dataset["label"]

    sst2_dataset = SST2Dataset(tokenized_texts, labels)
    dataloader = DataLoader(sst2_dataset, batch_size=batch_size)

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
            positive_prob = probs[:, 1].item()
            negative_prob = probs[:, 0].item()

        idx = len(data["index"])
        sentence = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        label = "positive" if labels[0].item() else "negative"

        data["index"].append(idx)
        data["sentence"].append(sentence)
        data["label"].append(label)
        data["positive_prob"].append(positive_prob)
        data["negative_prob"].append(negative_prob)

    pd.DataFrame(data).to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire(infer)
