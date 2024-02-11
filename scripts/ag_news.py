
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
    dataset = load_dataset('ag_news')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    tp_dict = { # acc: 0.2594736842105263
        0: "</E>\"</text>\" It is about world.",
        1: "</E>\"</text>\" It is about sports.",
        2: "</E>\"</text>\" It is about business.",
        3: "</E>\"</text>\" It is about science and technology.",
    }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    retriever = RandomRetriever(data, ice_num=8, seed=42, index_split='train', test_split='test')

    inferencer = PPLInferencer(
        model_name='distilgpt2',
        batch_size=1,
        output_json_filepath='iclx_output',
        output_json_filename='240111-sst5'
    )

    predictions = inferencer.inference(retriever, ice_template=template)
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)


if __name__ == '__main__':
    test()
