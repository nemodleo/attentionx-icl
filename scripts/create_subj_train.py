from datasets import Dataset, DatasetDict
from openicl import DatasetReader
import json
import vessl

from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer.parent_inferencer import ParentInferencer

vessl.init()

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
    0 : "</E>Review: </text>\nSubjectivity: Subjective",
    1 : "</E>Review: </text>\nSubjectivity: Objective",
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
retriever = RandomRetriever(data, ice_num=0)
inferencer = ParentInferencer(model_name='EleutherAI/gpt-j-6b')
predictions = inferencer.inference(retriever, ice_template=template)
