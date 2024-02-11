
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
    dataset = load_dataset('trec')
    dataset = dataset.rename_column('coarse_label', 'label')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    tp_dict = { # acc: 0.27
        0: "</E>\"</text>\" It is about abbreviation.",
        1: "</E>\"</text>\" It is about entity.",
        2: "</E>\"</text>\" It is about description and abstract concept.",
        3: "</E>\"</text>\" It is about human being.",
        4: "</E>\"</text>\" It is about location.",
        5: "</E>\"</text>\" It is about numeric value."
    }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

    retriever = RandomRetriever(data, ice_num=8, seed=42, index_split='train', test_split='test')

    inferencer = PPLInferencer(
        model_name='distilgpt2',
        batch_size=1,
        output_json_filepath='iclx_output',
        output_json_filename='240111-trec'
    )

    predictions = inferencer.inference(retriever, ice_template=template)
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)


if __name__ == '__main__':
    test()
