import sys
import os
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
from datasets import Dataset
from datasets import DatasetDict
from loguru import logger


def test(shots=10, model_name='EleutherAI/gpt-neo-2.7B', retriever=TopkRetriever, seed=42, batch_size=1):

    def gen(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                yield json.loads(line)

    train_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/train_sst5_bert.jsonl"})
    val_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/dev.jsonl"})
    test_ds = Dataset.from_generator(gen, gen_kwargs={"file_path": "data/sst5/test.jsonl"})

    # Define a DatasetReader, with specified column names where input and output are stored.
    dataset = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})
    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    naive, sequence, binning, gt, pseudo_gt = [], [], [], [], []
    s_1_1, s_1_2, s_2_1, s_2_2, s_2_3, s_3_1, s_3_2 = [], [], [], [], [], [], []
    b_1_1, b_2_1, b_2_2, b_2_3, b_3_1, b_3_2 = [], [], [], [], [], []

    instructions = "In this task, you'll analyze movie reviews and classify them into sentiment categories: \"great\", \"good\", \"okay\", \"bad\", or \"terrible\". Read each review, analyze its sentiment based on language, tone, and overall impression, then classify it accordingly. \"Great\" signifies overwhelmingly positive sentiment, \"good\" for positive overall, \"okay\" for neutral or mixed, \"bad\" for mainly negative, and \"terrible\" for intensely negative. Spit out the most likely label, out of five.\n"
    
    shot_list = [0, 4, 8, 16, 32]
    x = [n for n in range(len(shot_list))]

    for i in shot_list:
        s_1_1.append(test_sequence_1_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        s_1_2.append(test_sequence_1_2(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        
        s_2_1.append(test_sequence_2_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        s_2_2.append(test_sequence_2_2(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        s_2_3.append(test_sequence_2_3(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])

        s_3_1.append(test_sequence_3_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        s_3_2.append(test_sequence_3_2(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])

        b_1_1.append(test_binning_1_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        b_2_1.append(test_binning_2_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        b_2_2.append(test_binning_2_2(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        b_2_3.append(test_binning_2_3(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])

        b_3_1.append(test_binning_3_1(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        b_3_2.append(test_binning_3_2(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])

        naive.append(test_naive(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        sequence.append(test_sequence(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        binning.append(test_binning(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        gt.append(test_GT(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])
        pseudo_gt.append(test_pseudo_GT(i, data, model_name, retriever, seed, batch_size, instructions)['accuracy'])

    logger.info(s_1_1)
    logger.info(s_1_2)
    
    logger.info(s_2_1)
    logger.info(s_2_2)
    logger.info(s_2_3)

    logger.info(s_3_1)
    logger.info(s_3_2)

    logger.info(b_1_1)
    logger.info(b_2_1)
    logger.info(b_2_2)
    logger.info(b_2_3)

    logger.info(b_3_1)
    logger.info(b_3_2)

    logger.info(naive)
    logger.info(sequence)
    logger.info(binning)
    logger.info(gt)
    logger.info(pseudo_gt)


    plt.plot(x, s_1_1, label='s1.1')
    plt.plot(x, s_1_2, label='s1.2')
    
    plt.plot(x, s_2_1, label='s2.1')
    plt.plot(x, s_2_2, label='s2.2')
    plt.plot(x, s_2_3, label='s2.3')

    plt.plot(x, s_3_1, label='s3.1')
    plt.plot(x, s_3_2, label='s3.2')

    plt.plot(x, b_1_1, label='b1.1')
    plt.plot(x, b_2_1, label='b2.1')
    plt.plot(x, b_2_2, label='b2.2')
    plt.plot(x, b_2_3, label='b2.3')

    plt.plot(x, b_3_1, label='b3.1')
    plt.plot(x, b_3_2, label='b3.2')

    plt.plot(x, naive, label='naive')
    plt.plot(x, sequence, label='sequence')
    plt.plot(x, binning, label='binning')
    plt.plot(x, gt, label='gt')
    plt.plot(x, pseudo_gt, label='pseudo_gt')

    plt.legend()
    plt.savefig('iclx_output/sst5.png')


def test_naive(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: great </VP>%% good </P>%% okay </N>%% bad </NG>% terrible </VN>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '4': '</VP>',
        '3': '</P>',
        '2': '</N>',
        '1': '</NG>',
        '0': '</VN>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, instructions=instructions)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>',
        '1': '</2>',
        'Label2': '</Label2>',
        '2': '</3>',
        'Label3': '</Label3>',
        '3': '</4>',
        'Label4': '</Label4>',
        '4': '</5>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_1_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>',
        '1': '</2>',
        'Label2': '</Label2>',
        '2': '</3>',
        'Label3': '</Label3>',
        '3': '</4>',
        'Label4': '</Label4>',
        '4': '</5>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='tags')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_1_2(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Label2> </Label3> </Label4> </Label5>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_2_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_2_2(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '1': '</2>',
        'Label2': '</Label2>',
        '2': '</3>',
        'Label3': '</Label3>',
        '3': '</4>',
        'Label4': '</Label4>',
        '4': '</5>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_2_3(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label1> </1>% </Label1> </1>% </Label1> </1>% </Label1> </1>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_3_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>',
        '1': '</2>',
        'Label2': '</Label2>',
        '2': '</3>',
        'Label3': '</Label3>',
        '3': '</4>',
        'Label4': '</Label4>',
        '4': '</5>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='labels')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_sequence_3_2(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </1>% </Label2> </2>% </Label3> </3>% </Label4> </4>% </Label5> </5>%"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        '0': '</1>',
        'Label1': '</Label1>',
        '1': '</2>',
        'Label2': '</Label2>',
        '2': '</3>',
        'Label3': '</Label3>',
        '3': '</4>',
        'Label4': '</Label4>',
        '4': '</5>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='labels_except_first')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> is very likely, </Label2> is likely, </Label3> could be likely, </Label4> is not likely, </Label5> is not very likely"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>'
    }
    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_1_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Exp1>, </Label2> </Exp2>, </Label3> </Exp3>, </Label4> </Exp4>, </Label5> </Exp5>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>', 
        'Exp1' : '</Exp1>',
        'Exp2' : '</Exp2>',
        'Exp3' : '</Exp3>',
        'Exp4' : '</Exp4>',
        'Exp5' : '</Exp5>'
    }

    # Define binning dict
    binning_dict = {
        'Exp1' : 'is very likely',
        'Exp2' : 'is likely',
        'Exp3' : 'could be likely',
        'Exp4' : 'is not likely',
        'Exp5' : 'is not very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='tags')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_2_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Exp1>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Exp1' : '</Exp1>'
    }

    # Define binning dict
    binning_dict = {
        'Exp1' : 'is very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_2_2(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label2> </Exp2>, </Label3> </Exp3>, </Label4> </Exp4>, </Label5> </Exp5>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>', 
        'Exp2' : '</Exp2>',
        'Exp3' : '</Exp3>',
        'Exp4' : '</Exp4>',
        'Exp5' : '</Exp5>'
    }

    # Define binning dict
    binning_dict = {
        'Exp2' : 'is likely',
        'Exp3' : 'could be likely',
        'Exp4' : 'is not likely',
        'Exp5' : 'is not very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_2_3(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Exp1> </Label1> </Exp1> </Label1> </Exp1> </Label1> </Exp1> </Label1> </Exp1>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Exp1' : '</Exp1>'
    }

    # Define binning dict
    binning_dict = {
        'Exp1' : 'is very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_3_1(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Exp1>, </Label2> </Exp2>, </Label3> </Exp3>, </Label4> </Exp4>, </Label5> </Exp5>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>', 
        'Exp1' : '</Exp1>',
        'Exp2' : '</Exp2>',
        'Exp3' : '</Exp3>',
        'Exp4' : '</Exp4>',
        'Exp5' : '</Exp5>'
    }

    # Define binning dict
    binning_dict = {
        'Exp1' : 'is very likely',
        'Exp2' : 'is likely',
        'Exp3' : 'could be likely',
        'Exp4' : 'is not likely',
        'Exp5' : 'is not very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='labels')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_binning_3_2(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template
    ice_dict = "</E>Review: </text>\nSentiment: </Label1> </Exp1>, </Label2> </Exp2>, </Label3> </Exp3>, </Label4> </Exp4>, </Label5> </Exp5>"

    # Inference prompt template
    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    label_dict = {
        '0': "terrible",
        '1': "bad",
        '2': "okay",
        '3': "good",
        '4': "great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {
        'text': '</text>',
        'Label1': '</Label1>',
        'Label2': '</Label2>',
        'Label3': '</Label3>',
        'Label4': '</Label4>',
        'Label5': '</Label5>', 
        'Exp1' : '</Exp1>',
        'Exp2' : '</Exp2>',
        'Exp3' : '</Exp3>',
        'Exp4' : '</Exp4>',
        'Exp5' : '</Exp5>'
    }

    # Define binning dict
    binning_dict = {
        'Exp1' : 'is very likely',
        'Exp2' : 'is likely',
        'Exp3' : 'could be likely',
        'Exp4' : 'is not likely',
        'Exp5' : 'is not very likely'
    }

    ice_template = PromptTemplate(ice_dict, column_token_map, label_dict=label_dict, ice_token='</E>', binning=binning_dict)
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed, use_ordering=True)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, shuffle='labels_except_first')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_GT(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # ICL exemplar template & Inference prompt template
    ice_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {'text': '</text>'}
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score

def test_pseudo_GT(ice_num, data, model_name, retriever, seed, batch_size, instructions):

    # Inference prompt template
    ice_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    tp_dict = {
        '0': "</E>Review: </text>\nSentiment: terrible",
        '1': "</E>Review: </text>\nSentiment: bad",
        '2': "</E>Review: </text>\nSentiment: okay",
        '3': "</E>Review: </text>\nSentiment: good",
        '4': "</E>Review: </text>\nSentiment: great"
    }

    # Define prompt templates for ice and prompt
    column_token_map = {'text': '</text>'}
    ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
    prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = retriever(data, ice_num=ice_num, seed=seed)
    inferencer = PPLInferencer(model_name=model_name, labels=['0', '1', '2', '3', '4'], batch_size=batch_size, instructions=instructions)

    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template, pseudo_gt='pseudo_gt')
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)

    return score


if __name__ == '__main__':
    test()
