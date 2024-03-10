
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import PromptTemplate
from iclx import RandomRetriever
from iclx import PPLInferencer
from iclx import AccEvaluator
sys.path.pop()


def test():
    # 'SetFit/sst5'
    dataset = load_dataset('data/SetFit___sst5')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    ice_dict = {
        0: "</E>Movie Review: </text>\nVery Negative",
        1: "</E>Movie Review: </text>\nNegative",
        2: "</E>Movie Review: </text>\nNeutral",
        3: "</E>Movie Review: </text>\nPositive",
        4: "</E>Movie Review: </text>\nVery Positive"
    }
    ice_template = PromptTemplate(ice_dict, {'text': '</text>'}, ice_token='</E>')

    prompt_dict = {
        0: "</E>Movie Review: </text>\nVery Negative",
        1: "</E>Movie Review: </text>\nNegative",
        2: "</E>Movie Review: </text>\nNeutral",
        3: "</E>Movie Review: </text>\nPositive",
        4: "</E>Movie Review: </text>\nVery Positive"
    }
    prompt_template = PromptTemplate(prompt_dict, {'text': '</text>'}, ice_token='</E>')

    retriever = RandomRetriever(data, ice_num=8, seed=42, index_split='train', test_split='test')

    # 'distilgpt2'
    inferencer = PPLInferencer(
        model_name='ckpt/models--distilgpt2',
        batch_size=1,
        output_json_filepath='iclx_output',
        output_json_filename='240111-sst5'
    )

    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)


if __name__ == '__main__':
    test()
