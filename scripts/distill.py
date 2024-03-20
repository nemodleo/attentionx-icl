from datasets import Dataset, DatasetDict
import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx import TopkRetriever
from iclx import PPLInferencer
from iclx import AccEvaluator
sys.path.pop()

import matplotlib.pyplot as plt
import json
import vessl 
vessl.init()

retriever_dict = {"TopK": TopkRetriever,
                "Random": RandomRetriever}


def test(shots=10, model_name='distilgpt2', retriever=RandomRetriever, seed=42):
    print(retriever)
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
        naive.append(test_naive(i, data, model_name, retriever, seed)['accuracy'])
        sequence.append(test_sequence(i, data, model_name, retriever, seed)['accuracy'])
        binning.append(test_binning(i, data, model_name, retriever, seed)['accuracy'])
        gt.append(test_GT(i, data, model_name, retriever, seed)['accuracy'])
        pseudo_gt.append(test_pseudo_GT(i, data, model_name, retriever, seed)['accuracy'])

    print(naive)
    print(sequence)
    print(binning)
    print(gt)
    print(pseudo_gt)

    plt.plot(x, naive, label = 'naive')
    plt.plot(x, sequence, label = 'sequence')
    plt.plot(x, binning, label = 'binning')
    plt.plot(x, gt, label = 'gt')
    plt.plot(x, pseudo_gt, label = 'pseudo_gt')

    plt.legend()
    plt.savefig(OUTPUT_PATH)


def test_naive(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = ICE_DICT["naive"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_MAP

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["naive"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, labels= LABELS)
    inferencer = PPLInferencer(model_name=model_name, labels= LABELS)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_sequence(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = ICE_DICT["sequence"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_MAP

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["sequence"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, labels=LABELS, order=True)
    inferencer = PPLInferencer(model_name=model_name, labels=LABELS)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = ICE_DICT["binning"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_MAP

    column_token_map = COLUMN_TOKEN_MAP["binning"]

    # Define prompt templates for ice and prompt
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, labels=LABELS, order=True)  
    inferencer = PPLInferencer(model_name=model_name, labels=LABELS)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_GT(ice_num, data, model_name, retriever, seed):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, labels= LABELS)
    inferencer = PPLInferencer(model_name=model_name, labels=LABELS)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_pseudo_GT(ice_num, data, model_name, retriever, seed):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, labels=LABELS)
    inferencer = PPLInferencer(model_name=model_name, labels=LABELS)

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

    RETRIEVER = retriever_dict[setup['retriever']]
    TRAIN_PATH = setup['train_path']
    VAL_PATH = setup['val_path']
    TEST_PATH = setup['test_path']
    
    
    DATA_COLUMNS = setup['data_columns']
    ICE_DICT = setup['ice_dict']
    TP_DICT = setup['template_dict']
    
    LABELS = setup['labels']
    LABEL_MAP = setup['label_map']
    COLUMN_TOKEN_MAP = setup['column_token_map']
    
    OUTPUT_PATH = setup['output_path']
    
    test(retriever=RETRIEVER)