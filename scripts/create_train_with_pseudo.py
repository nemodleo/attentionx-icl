from datasets import Dataset, DatasetDict
import sys
import os
from loguru import logger
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json

from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx.inferencer.parent_inferencer import ParentInferencer

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
    
    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TRAIN_PATH})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TRAIN_PATH})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TRAIN_PATH})
    
    dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    dataset = dataset_dict
    
    data = DatasetReader(dataset, input_columns=DATA_COLUMNS['input_columns'], output_column=DATA_COLUMNS['output_columns'][0])
    
    
    template = PromptTemplate(TP_DICT, {'text': '</text>'}, ice_token='</E>')
    retriever = RandomRetriever(data, ice_num=0)
    inferencer = ParentInferencer(model_name=TEACHER, batch_size=BATCH_SIZE)
    predictions = inferencer.inference(retriever, ice_template=template)
    
    for i, p in enumerate(predictions):
        entry = dataset_dict["test"][i]
        p["text"] = entry["text"] 
        p["label"] = str(entry["label"])
        p["label_text"] = label_map[p["label"]]

        label_keys = list(LABEL_MAP.keys())
        perplexity_values = [p[int(k)] for k in label_keys]      
        probabilities = rec_softmax(perplexity_values)
        max_label_index = int(np.argmax(probabilities))
        p["pseudo_gt"] = str(max_label_index)

        for k in label_keys:
            p.pop(k, None)
        p.update({str(k): v for k, v in zip(range(len(label_keys)), probabilities)})
    
    # Save predictions as file ! 
    with open(OUTPUT_PATH, 'w') as f:
        for entry in predictions:
            json.dump(entry, f)
            f.write('\n')
    logger.info("Finished saving the data to OUTPUT_PATH")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('setup_dict', type=str, help='Path to the setup dictionary json file')
    args = parser.parse_args()

    setup = json.load(open(args.setup_dict, 'r'))
    
    TRAIN_PATH = setup['train_path']
    OUTPUT_PATH = setup['output_path']
    
    TP_DICT = setup['template_dict']
    DATA_COLUMNS = setup['data_columns']
    LABEL_MAP = setup['label_map']
    
    BATCH_SIZE = setup['batch_size']
    TEACHER = setup['teacher_model']

    logger.info(f"Creating train data with PseudoGT Label using teacher model {TEACHER}")
    logger.info(f"Using train data from: {TRAIN_PATH}")
    logger.info(f"Created data will be saved to {OUTPUT_PATH}")

    create_data()
