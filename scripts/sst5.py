
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
    dataset = load_dataset('SetFit/sst5')

    data = DatasetReader(dataset, input_columns=['text'], output_column='label')

    tp_dict = { # acc: 0.18823529411764706
        0: "</E>Movie Review: </text> \n Very Negative",
        1: "</E>Movie Review: </text> \n Negative",
        2: "</E>Movie Review: </text> \n Neutral",
        3: "</E>Movie Review: </text> \n Positive",
        4: "</E>Movie Review: </text> \n Very Positive"
    }
    # tp_dict = { # acc: 0.269683257918552
    #     0: "</E>Movie Review: </text>\nVery Negative",
    #     1: "</E>Movie Review: </text>\nNegative",
    #     2: "</E>Movie Review: </text>\nNeutral",
    #     3: "</E>Movie Review: </text>\nPositive",
    #     4: "</E>Movie Review: </text>\nVery Positive"
    # }
    # tp_dict = { # acc: 0.18461538461538463
    #     0: "</E>Movie Review: </text>\nSentiment: Very Negative",
    #     1: "</E>Movie Review: </text>\nSentiment: Negative",
    #     2: "</E>Movie Review: </text>\nSentiment: Neutral",
    #     3: "</E>Movie Review: </text>\nSentiment: Positive",
    #     4: "</E>Movie Review: </text>\nSentiment: Very Positive",
    # }
    # tp_dict = { # acc: 0.26199095022624436
    #     0: "</E>Movie Review: </text>\nSentiment: terrible",
    #     1: "</E>Movie Review: </text>\nSentiment: bad",
    #     2: "</E>Movie Review: </text>\nSentiment: okay",
    #     3: "</E>Movie Review: </text>\nSentiment: good",
    #     4: "</E>Movie Review: </text>\nSentiment: great",
    # }

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
