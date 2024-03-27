
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import TopkRetriever
from iclx import PPLInferencer
from iclx import AccEvaluator
sys.path.pop()


def test():
    dataset = load_dataset('gpt3mix/sst2')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    ice_dict = {
        0: "</E>Movie Review: </text>\nPositive",
        1: "</E>Movie Review: </text>\nNegative"
    }
    ice_template = PromptTemplate(ice_dict, {'text': '</text>'}, ice_token='</E>')


    prompt_dict = {
        0: "</E>Movie Review: </text>\nPositive",
        1: "</E>Movie Review: </text>\nNegative"
    }
    prompt_template = PromptTemplate(prompt_dict, {'text': '</text>'}, ice_token='</E>')

    retriever = TopkRetriever(
        data, ice_num=8, index_split='train', test_split='test',
        sentence_transformers_model_name='all-mpnet-base-v2',
        tokenizer_name='gpt2-xl'
    )

    inferencer = PPLInferencer(
        model_name='distilgpt2',
        batch_size=1,
        output_json_filepath='iclx_output',
        output_json_filename='240310-sst2-topk'
    )

    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)


if __name__ == '__main__':
    test()
