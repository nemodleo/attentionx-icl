import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx import PPLInferencer
from iclx import AccEvaluator
sys.path.pop()
import matplotlib.pyplot as plt
import json
import vessl
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger


def test(shots=10, model_name='distilgpt2', retriever=RandomRetriever, seed=42):

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst2/train_label_sst2.jsonl"})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst2/dev.jsonl"})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst2/test.jsonl"})

    dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    data = DatasetReader(dataset, input_columns=['text'], output_column= 'label')

    naive, sequence, binning, gt, pseudo_gt = [], [], [], [], []
    x = [n for n in range(shots)]

    for i in range(shots):
        naive.append(test_naive(i, data, model_name, retriever, seed)['accuracy'])
        sequence.append(test_sequence(i, data, model_name, retriever, seed)['accuracy'])
        binning.append(test_binning(i, data, model_name, retriever, seed)['accuracy'])
        gt.append(test_GT(i, data, model_name, retriever, seed)['accuracy'])
        pseudo_gt.append(test_pseudo_GT(i, data, model_name, retriever, seed)['accuracy'])

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
    plt.savefig('/output/sst2.png')


def test_naive(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: Positive </P>% Negative </N>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    label_dict = {
        '0': "Negative",
        '1': "Positive"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '1': '</P>',
        '0': '</N>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1'])

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_sequence(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label2> </2>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    label_dict = {
        '0': "Negative",
        '1': "Positive"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>',
        '1': '</2>',
        'Label2': '</Label2>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1'])

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_binning(ice_num, data, model_name, retriever, seed):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> is very likely, </Label2> is not very likely"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    label_dict = {
        '0': "Negative",
        '1': "Positive"
    }

    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>'
    }

    # Define prompt templates for ice and prompt
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)  
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1'])

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_GT(ice_num, data, model_name, retriever, seed):

    # Inference prompt template
    ice_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {'text': '</text>'}
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1'])

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


def test_pseudo_GT(ice_num, data, model_name, retriever, seed):

    # Inference prompt template
    ice_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: Negative",
        '1': "</E>Review: </text>\nSentiment: Positive"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {'text': '</text>'}
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1'])

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, pseudo_gt='pseudo_gt')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


if __name__ == '__main__':
    vessl.init()
    test()
