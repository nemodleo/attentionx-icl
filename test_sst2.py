# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from datasets import Dataset, DatasetDict
from openicl import DatasetReader
import json
import pandas as pd

def gen(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)


def gen_pd(file_path):
    df = pd.read_csv(file_path)
    for idx, row in df.iterrows():
        yield row.to_dict()
            
train_ds = Dataset.from_generator(gen_pd, gen_kwargs={"file_path": "data/sst2/train2.csv"})
val_ds = Dataset.from_generator(gen_pd, gen_kwargs={"file_path": "data/sst2/train2.csv"})
test_ds = Dataset.from_generator(gen_pd, gen_kwargs={"file_path": "data/sst2/test2.csv"})

dataset_dict = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})


# Loading dataset from huggingface
# lets do sst5
#dataset = load_dataset('SetFit/sst5')
dataset = dataset_dict 

print(dataset['train'])
print(dataset['test']['label'])

# Define a DatasetReader, with specified column names where input and output are stored.
# TODO : Define a DatasetReader 
data = DatasetReader(dataset, input_columns=['text'], output_column= 'label')

print(dataset.keys())  # prints the names of the available splits
train_dataset = dataset['train']  # gets the training split
test_dataset = dataset['test']  # gets the testing split

# TODO : Alter PromptTemplate 
from openicl import PromptTemplate

# need to make them show percentage
ice_dict = "</E> Movie Review: </text> \n Positive </P>% Negative </N>%"

ice_dict2 = {
    0 : "</E>Movie Review: </text> \n Negative",
    1 : "</E>Movie Review: </text> \n Positive",
}

tp_dict = {
    0 : "</E>Negative Movie Review: </text>",
    1 : "</E>Positive Movie Review: </text>",
}

column_token_map = {'text': '</text>', 'positive_prob' : '</P>', 'negative_prob' : '</N>' }
ice_template = PromptTemplate(ice_dict, column_token_map, ice_token='</E>')
prompt_template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')


from openicl import RandomRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = RandomRetriever(data, ice_num=2, labels= [0,1] )

from openicl import PPLInferencer
inferencer = PPLInferencer(model_name='distilgpt2', labels= [0,1])

from openicl import AccEvaluator
# the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
predictions = inferencer.inference(retriever, ice_template=ice_template, prompt_template=prompt_template)
# compute accuracy for the prediction
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
