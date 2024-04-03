import torch
from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from datasets import Dataset
from datasets import DatasetDict
from transformers import AutoTokenizer


class DatasetReader:
    """In-conext Learning Dataset Reader Class
        Generate an DatasetReader instance through 'dataset'.

    Attributes:
        dataset (:obj:`Dataset` or :obj:`DatasetDict`): The dataset to be read.
        input_columns (:obj:`List[str]` or :obj:`str`): A list of column names (a string of column name) in the dataset that represent(s) the input field.
        output_column (:obj:`str`): A column name in the dataset that represents the prediction field.
        ds_size (:obj:`int` or :obj:`float`, optional): The number of pieces of data to return. When ds_size is an integer and greater than or equal to 1, `ds_size` pieces of data are randomly returned. When 0 < :obj:`ds_size` < 1, ``int(len(dataset) * ds_size)`` pieces of data are randomly returned. (used for testing)
        references(:obj:`list`, optional): The list of references, initialized by ``self.dataset[self.test_split][self.output_column]``.
        input_template (:obj:`PromptTemplate`, optional): An instance of the :obj:`PromptTemplate` class, used to format the input field content during the retrieval process. (in some retrieval methods)
        output_template (:obj:`PromptTemplate`, optional): An instance of the :obj:`PromptTemplate` class, used to format the output field content during the retrieval process. (in some learnable retrieval methods)
        input_output_template (:obj:`PromptTemplate`, optional): An instance of the `PromptTemplate` class, used to format the input-output field content during the retrieval process. (in some retrieval methods)
    """

    def __init__(self,
                 dataset: Union[Dataset, DatasetDict],
                 input_columns: List[str],
                 output_column: str,
                 test_split: Optional[str] = 'test',
                 **kwargs,
                 ) -> None:
        self.input_columns = input_columns
        self.output_column = output_column
        self.dataset = dataset
        if isinstance(self.dataset, DatasetDict):
            if test_split in self.dataset.keys():
                self.references = self.dataset[test_split][self.output_column]
        elif isinstance(self.dataset, Dataset):
            self.references = self.dataset[self.output_column]

    def generate_input_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the input field based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = ' '.join([str(entry[ctx]) for ctx in self.input_columns])
        return prompt

    def generate_input_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None) -> List[
        str]:
        """Generate corpus for input field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict` instance.
            split (:obj:`str`, optional): The split of the dataset to use. If :obj:`None`, the entire dataset will be used. Defaults to ``None``.

        Returns:
            :obj:`List[str]`: A list of generated input field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_input_field_prompt(entry))
        return corpus


class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt', verbose=False)
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                             "text": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]
