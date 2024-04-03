
from datasets import load_dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from iclx import DatasetReader
from iclx import ProbPromptTemplate
from iclx import RandomRetriever
from iclx import ProbInferencer
from iclx import AccEvaluator
sys.path.pop()


def test():
    dataset = load_dataset('gpt3mix/sst2')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    ice_dict = {
        "0": "Positive",
        "1": "Negative"
    }
    prefix_template = "</E></text> It is"
    concat_token = "\n"

    ice_template = ProbPromptTemplate(prefix_template, ice_dict, concat_token, {'text': '</text>'}, ice_token='</E>')


    prompt_dict = {
        "0": "Positive",
        "1": "Negative"
    }
    prompt_template = ProbPromptTemplate(prefix_template, prompt_dict, concat_token, {'text': '</text>'}, ice_token='</E>')

    retriever = RandomRetriever(data, ice_num=8, seed=42, index_split='train', test_split='test')

    inferencer = ProbInferencer(
        model_name='distilgpt2',
        batch_size=1,
        output_json_filepath='iclx_output',
        output_json_filename='240110-sst2'
    )

    predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)


if __name__ == '__main__':
    test()