from datasets import Dataset, DatasetDict
import sys
import os
from loguru import logger
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import json

from iclx import DatasetReader
from iclx import ProbPromptTemplate
from iclx import ProbInferencer
from iclx import RandomRetriever

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
    
    
    template = ProbPromptTemplate(PREFIX_TP, LABEL_MAP, CONCAT_TOKEN, {'text': '</text>'}, ice_token='</E>')
    retriever = RandomRetriever(data, ice_num=8)
    inferencer = ProbInferencer(model_name=TEACHER, batch_size=BATCH_SIZE)
    probs, predictions = inferencer.inference(retriever, ice_template=template, prompt_template=template)
    
    teacher_data = []
    for i, (prob, pred) in enumerate(zip(probs, predictions)):
        new_entry = dict()
        entry = dataset_dict["test"][i]
        new_entry["text"] = entry["text"] 
        new_entry["label"] = str(entry["label"])
        new_entry["label_text"] = LABEL_MAP[str(entry["label"])]

        label_keys = list(LABEL_MAP.keys())
        new_entry["pseudo_gt"] = str(pred)
        prob = list(np.array(prob) / np.sum(prob))

        new_entry.update({str(k): v for k, v in zip(range(len(label_keys)), prob)})
        teacher_data.append(new_entry)

    # Save predictions as file ! 
    with open(OUTPUT_PATH, 'w') as f:
        for entry in teacher_data:
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
    
    PREFIX_TP = setup['prefix_template']
    CONCAT_TOKEN = setup['concat_token']
    DATA_COLUMNS = setup['data_columns']
    LABEL_MAP = setup['label_map']
    
    BATCH_SIZE = setup['batch_size']
    TEACHER = setup['teacher_model']

    logger.info(f"Creating train data with PseudoGT Label using teacher model {TEACHER}")
    logger.info(f"Using train data from: {TRAIN_PATH}")
    logger.info(f"Created data will be saved to {OUTPUT_PATH}")

    create_data()
