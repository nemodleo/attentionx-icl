import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer import ParentInferencer
sys.path.pop()
import json
import vessl
from datasets import Dataset
from datasets import DatasetDict
import numpy as np


def rec_softmax(x):
    e_x = np.exp(x)
    e_x = 1/e_x
    return e_x / e_x.sum(axis=0)


def create_data():

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/subj/train.jsonl"})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/subj/train.jsonl"})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/subj/train.jsonl"})

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    dataset = dataset_dict

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    tp_dict = {
        0: "</E>Review: </text>\nSubjectivity: Subjective",
        1: "</E>Review: </text>\nSubjectivity: Objective",
    }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
    retriever = RandomRetriever(data, ice_num=0)
    inferencer = ParentInferencer(model_name='EleutherAI/gpt-j-6B')
    predictions = inferencer.inference(retriever, ice_template=template)

    for i, p in enumerate(predictions):
        p["text"] = dataset_dict["test"][i]["text"] 
        p["label"] = dataset_dict["test"][i]["label"] 
        p["label_text"] = "subjective" if dataset_dict["test"][i]["label"] == 0 else "objective"

        perplexity_values = [p[0], p[1]]
        probabilities = rec_softmax(perplexity_values)

        p.pop(0, None)
        p.pop(1, None)
        p.update({str(k): v for k, v in zip(range(2), probabilities)})

    # Save predictions as file
    with open('data/subj/train_new_subj.jsonl', 'w') as f:
        for entry in predictions:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    vessl.init()
    create_data()
