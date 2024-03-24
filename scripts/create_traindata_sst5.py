from datasets import Dataset, DatasetDict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer.parent_inferencer import ParentInferencer
sys.path.pop()

import matplotlib.pyplot as plt
import json
import vessl 
vessl.init()

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

print(dataset.keys())  # prints the names of the available splits
train_dataset = dataset['train']  # gets the training split
test_dataset = dataset['test']  # gets the testing split


'''
tp_dict = {
    0 : "</E>Movie Review: </text> Very Negative",
    1 : "</E>Movie Review: </text> Negative",
    2 : "</E>Movie Review: </text> Neutral" ,
    3 : "</E>Movie Review: </text> Positive" ,
    4 : "</E>Movie Review: </text> Very Positive" 
}
'''

tp_dict = {
    0 : "</E>Review: </text>\nSentiment: terrible",
    1 : "</E>Review: </text>\nSentiment: bad",
    2 : "</E>Review: </text>\nSentiment: okay" ,
    3 : "</E>Review: </text>\nSentiment: good" ,
    4 : "</E>Review: </text>\nSentiment: great" 
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = RandomRetriever(data, ice_num=0)

inferencer = ParentInferencer.ParentInferencer(model_name='EleutherAI/gpt-j-6B')

# the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
predictions = inferencer.inference(retriever, ice_template=template)

for i, p in enumerate(predictions):
    p["text"] = dataset_dict["test"][i]["text"]


# Save predictions as file ! 
with open('/output/train_label_sst5.jsonl', 'w') as f:
    for entry in predictions:
        json.dump(entry, f)
        f.write('\n')