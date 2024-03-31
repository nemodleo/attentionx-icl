import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer import ParentInferencer
sys.path.pop()
import json
from datasets import Dataset
from datasets import DatasetDict


def create_data():

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/train_sst5.jsonl"})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/train_sst5.jsonl"})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/train_sst5.jsonl"})

    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    dataset = dataset_dict

    # Define a DatasetReader, with specified column names where input and output are stored.
    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    # Inference Prompt template.
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = RandomRetriever(data, ice_num=0)

    # class ParentInferencer is modified to spit soft labels as predictions
    inferencer = ParentInferencer.ParentInferencer(model_name='EleutherAI/gpt-j-6B')

    predictions = inferencer.inference(retriever, ice_template=template)

    for i, p in enumerate(predictions):
        p["text"] = dataset_dict["test"][i]["text"]

    # Save predictions as file
    with open('data/sst5/train_label_sst5.jsonl', 'w') as f:
        for entry in predictions:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    create_data()
