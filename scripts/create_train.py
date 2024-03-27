from datasets import Dataset, DatasetDict
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json
import vessl

from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer.parent_inferencer import ParentInferencer

import numpy as np

def rec_softmax(x):
    e_x = np.exp(x)
    e_x = 1/e_x
    return e_x / e_x.sum(axis=0)


vessl.init()

def create_data(setup_dict):
    setup = json.load(open(setup_dict, 'r'))
    train_path = setup['train_path']
    tp_dict = setup['template_dict']
    data_columns = setup['data_columns']
    label_map = setup['label_map']
    output_path = setup['output_path']
    # print(train_path, tp_dict, output_path)
    
    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)
    
    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": train_path})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": train_path})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": train_path})
    
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    dataset = dataset_dict
    
    data = DatasetReader(dataset, input_columns=data_columns['input_columns'], output_column=data_columns['output_columns'][0])
    
    
    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
    retriever = RandomRetriever(data, ice_num=0)
    inferencer = ParentInferencer(model_name='EleutherAI/gpt-j-6B', batch_size=1)
    predictions = inferencer.inference(retriever, ice_template=template)
    
    for i, p in enumerate(predictions):
        entry = dataset_dict["test"][i]
        p["text"] = entry["text"] 
        p["label"] = str(entry["label"])
        p["label_text"] = label_map[p["label"]]

        label_keys = list(label_map.keys())
        perplexity_values = [p[int(k)] for k in label_keys]      
        probabilities = rec_softmax(perplexity_values)
        max_label_index = int(np.argmax(probabilities))
        p["pseudo_gt"] = str(max_label_index)

        for k in label_keys:
            p.pop(k, None)
        p.update({str(k): v for k, v in zip(range(len(label_keys)), probabilities)})
    
    # Save predictions as file ! 
    with open(output_path, 'w') as f:
        for entry in predictions:
            json.dump(entry, f)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('setup_dict', type=str, help='Path to the setup dictionary json file')
    args = parser.parse_args()

    create_data(args.setup_dict)
