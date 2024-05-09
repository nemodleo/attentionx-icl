import sys
import os
import gc
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
import torch

retriever_dict = {"topk": TopkRetriever,
                "random": RandomRetriever}

def clean_up_memory():
    gc.collect()
    torch.cuda.empty_cache()

def test(shots=[32, 16, 8, 4, 2, 1], model_name='distilgpt2', retriever_cls=RandomRetriever, retriever_base='all-mpnet-base-v2', batch_size=1):
    assert all(shots[i] > shots[i+1] for i in range(len(shots)-1)), "Shots should be in descending order"

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TRAIN_PATH})
    val_ds = None if not VAL_PATH else Dataset.from_generator(gen, gen_kwargs={"file_path": VAL_PATH})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": TEST_PATH})


    dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    data = DatasetReader(dataset, input_columns=DATA_COLUMNS['input_columns'], output_column=DATA_COLUMNS['output_columns'][0])


    sequence, binning, gt, pseudo_gt, seq_extreme, seq_uniform = [], [], [], [], [], []

    with open(f"{FOLDER_NAME}/acc_{EXP_NAME}.txt", 'a') as f:
        f.write("sequence, binning, gt, pseudo_gt, seq_extreme, seq_uniform\n")

        retriever = retriever_cls(data, sentence_transformers_model_name=retriever_base, ice_num=shots[0])

        # number of shots to run
        for i in shots:
            logger.info(f"Running for shot {i}")
            retriever.ice_num = i

            sequence.append(test_sequence(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"sequence for shot {i} done")

            binning.append(test_binning(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"binning for shot {i} done")

            gt.append(test_GT(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"gt for shot {i} done")

            pseudo_gt.append(test_pseudo_GT(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"pseudo_gt for shot {i} done")

            # sequence ablation 
            seq_extreme.append(test_seq_extreme(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"seq_extreme for shot {i} done")

            seq_uniform.append(test_seq_uniform(data, model_name, retriever, batch_size)['accuracy'])
            clean_up_memory()
            logger.info(f"seq_uniform for shot {i} done")

            f.write(f"{sequence[-1]}, {binning[-1]}, {gt[-1]}, {pseudo_gt[-1]}, {seq_extreme[-1]}, {seq_uniform[-1]}\n")
            f.flush()
            logger.info(f"Finished logging accuracies for {i} shot")

    logger.info(sequence)
    logger.info(binning)
    logger.info(gt)
    logger.info(pseudo_gt)
    logger.info(seq_extreme)
    logger.info(seq_uniform)

    # Plotting in reverse order
    xticks = range(len(shots))
    plt.plot(xticks, sequence[::-1], label='sequence')
    plt.plot(xticks, binning[::-1], label='binning')
    plt.plot(xticks, gt[::-1], label='gt')
    plt.plot(xticks, pseudo_gt[::-1], label='pseudo_gt')
    plt.plot(xticks, seq_extreme[::-1], label='seq_extreme')
    plt.plot(xticks, seq_uniform[::-1], label='seq_uniform')
    plt.xticks(xticks, shots[::-1])

    plt.legend()
    plt.savefig(f"{FOLDER_NAME}/plot_{EXP_NAME}.png")

    logger.info(f"Finished running and saving artifacts for experiment {EXP_NAME}")

def test_seq_extreme(data, model_name, retriever, batch_size):

    # ICL exemplar template
    ice_dict = ICE_DICT["seq_extreme"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["sequence"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)


    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=True)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_seq_uniform(data, model_name, retriever, batch_size):

    # ICL exemplar template
    ice_dict = ICE_DICT["seq_uniform"]

    # Inference prompt template
    tp_dict = TP_DICT

    label_dict = LABEL_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["sequence"]
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)


    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=True)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence(data, model_name, retriever, batch_size):

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
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)


    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=True)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_binning(data, model_name, retriever, batch_size):

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
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=True)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_GT(data, model_name, retriever, batch_size):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=False)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_pseudo_GT(data, model_name, retriever, batch_size):

    # Inference prompt template
    ice_dict = TP_DICT

    tp_dict = TP_DICT

    # Define prompt templates for ice and prompt
    column_token_map = COLUMN_TOKEN_MAP["GT"]
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    inferencer = PPLInferencer(model_name=model_name,
                               labels=list(LABEL_DICT.keys()),
                               batch_size=batch_size,
                               task_description=TASK_DESC)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, use_ordering=False, pseudo_gt='pseudo_gt')
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
    RETRIEVER_BASE = setup['retriever_base']
    STUDENT = setup['student']
    SHOTS = setup['shots']

    TRAIN_PATH = setup['train_path']
    VAL_PATH = setup['val_path']
    TEST_PATH = setup['test_path']
    TASK_DESC = setup['task_description']
    logger.info(f"Your task description:\n{TASK_DESC}")

    DATA_COLUMNS = setup['data_columns']
    ICE_DICT = setup['ice_dict']
    TP_DICT = setup['template_dict']

    LABEL_DICT = setup['label_dict']
    COLUMN_TOKEN_MAP = setup['column_token_map']

    EXP_NAME = setup['experiment_name']
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    FOLDER_NAME = f"output/{now}_{EXP_NAME}"

    os.makedirs(FOLDER_NAME, exist_ok=True)

    logger.info(f"Experiment: {EXP_NAME}")
    logger.info(f"Starting distillation of {SHOTS} shots using {STUDENT} student with {setup['retriever']} retriever.")
    logger.info(f"Using training data from {TRAIN_PATH}")
    logger.info(f"output will be saved to {FOLDER_NAME}")

    test(shots=SHOTS, model_name = STUDENT, retriever_cls=RETRIEVER, retriever_base=RETRIEVER_BASE, batch_size=BATCH_SIZE)
