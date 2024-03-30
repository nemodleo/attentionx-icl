import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever, TopkRetriever
from iclx import PPLInferencer
from iclx import AccEvaluator
sys.path.pop()
import matplotlib.pyplot as plt
import json
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger
from datetime import datetime
import argparse

retriever_dict = {"Topk": TopkRetriever,
                "Random": RandomRetriever}


def test(output_path, shots=10, model_name='distilgpt2', retriever=RandomRetriever, batch_size = 1):

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)
                
    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TRAIN_PATH})
    val_ds = None if not VAL_PATH else Dataset.from_generator(gen, gen_kwargs={"file_path": VAL_PATH})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TEST_PATH})

    dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    data = DatasetReader(dataset, input_columns=DATA_COLUMNS['input_columns'], output_column=DATA_COLUMNS['output_columns'][0])

    naive, sequence, binning, gt, pseudo_gt = [], [], [], [], []
    x = [n for n in range(shots)]

    for i in range(shots):
        naive.append(test_naive(i, data, model_name, retriever, batch_size)['accuracy'])
        sequence.append(test_sequence(i, data, model_name, retriever, batch_size)['accuracy'])
        binning.append(test_binning(i, data, model_name, retriever, batch_size)['accuracy'])
        gt.append(test_GT(i, data, model_name, retriever, batch_size)['accuracy'])
        pseudo_gt.append(test_pseudo_GT(i, data, model_name, retriever, batch_size)['accuracy'])

    logger.info(naive)
    logger.info(sequence)
    logger.info(binning)
    logger.info(gt)
    logger.info(pseudo_gt)

    plt.plot(x, naive, label = 'naive')
    plt.plot(x, sequence, label = 'sequence')
    plt.plot(x, binning, label = 'binning')
    plt.plot(x, gt, label = 'gt')
    plt.plot(x, pseudo_gt, label = 'pseudo_gt')

    plt.legend()
    plt.savefig(f"{output_path}/plot.png")

    accs = {'naive': naive, 'sequence': sequence, 'binning': binning, 'gt': gt, 'pseudo_gt': pseudo_gt}
    with open(f"{output_path}/acc.txt", 'w') as f:
        f.write(f"{output_path.split('/')[-1]}\n total shots: {shots}\n\n")
        for key, val in accs.items():
            f.write(f"{key}: {', '.joing(map(str,values))}\n")

def test_naive(ice_num, data, model_name, retriever, batch_size):

    # ICL exemplar template
    ice_dict = ICE_DICT["naive"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["naive"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num)
    inferencer = PPLInferencer(model_name=model_name, labels=list(LABEL_DICT.keys()), batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_sequence(ice_num, data, model_name, retriever, batch_size):

    # ICL exemplar template
    ice_dict = ICE_DICT["sequence"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["sequence"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=list(LABEL_DICT.keys()), batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning(ice_num, data, model_name, retriever, batch_size):

    # ICL exemplar template
    ice_dict = ICE_DICT["binning"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_DICT

    column_token_map = COLUMN_TOKEN_MAP["binning"]

    # Define prompt templates for ice and prompt
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, use_ordering=True)  
    inferencer = PPLInferencer(model_name=model_name, labels=list(LABEL_DICT.keys()), batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_GT(ice_num, data, model_name, retriever, batch_size):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num)
    inferencer = PPLInferencer(model_name=model_name, labels=list(LABEL_DICT.keys()), batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_pseudo_GT(ice_num, data, model_name, retriever, batch_size):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num)
    inferencer = PPLInferencer(model_name=model_name, labels=list(LABEL_DICT.keys()), batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, pseudo_gt='pseudo_gt')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('setup_dict', type=str, help='Path to the setup dictionary json file')
    args = parser.parse_args()

    setup = json.load(open(args.setup_dict, 'r'))

    BATCH_SIZE = setup['batch_size']
    RETRIEVER = retriever_dict[setup['retriever']]
    STUDENT = setup['student']
    SHOT_NUM = setup['shot_num']

    
    TRAIN_PATH = setup['train_path']
    VAL_PATH = setup['val_path']
    TEST_PATH = setup['test_path']
    
    
    DATA_COLUMNS = setup['data_columns']
    ICE_DICT = setup['ice_dict']
    TP_DICT = setup['template_dict']
    
    LABEL_DICT = setup['label_dict']
    COLUMN_TOKEN_MAP = setup['column_token_map']
    
    EXP_NAME = setup['experiment_name']
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"output/{now}_{EXP_NAME}"

    os.makedirs(folder_name, exist_ok=True)
    
    test(output_path = folder_name, shots=SHOT_NUM, model_name=STUDENT, retriever=RETRIEVER, batch_size=BATCH_SIZE)