"""
A script for generating inference data of fine-tuned model

Example Usage:
    python3 create_traindata_ft.py \
        --data_path ../data/sst5/train_sst5.jsonl \
        --output_path .../data/sst5/train_sst5_bert.jsonl \
        --model_name_or_path SetFit/distilbert-base-uncased__sst5__all-train
"""

import argparse
import json

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--data_col", type=str, default="text")
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path).to(args.device)
    tkn = AutoTokenizer.from_pretrained(args.model_name_or_path)

    data = []
    with open(args.data_path, "r") as f:
        while line := f.readline():
            data.append(json.loads(line))


    for item in tqdm(data):
        text = item["text"]
        tokenized = tkn([text])
        tokenized = {k: torch.tensor(tokenized[k]).to(args.device) for k in tokenized.keys()}
        probs = F.softmax(model(tokenized["input_ids"]).logits, dim=-1)
        item["pseudo_gt"] = torch.argmax(probs[0]).item()
        for i in range(probs.shape[1]):
            item[str(i)] = probs[0][i].item()

    with open(args.data_path, "w") as f:
        for row in data:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
