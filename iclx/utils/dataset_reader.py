from typing import List
from typing import Union
from typing import Optional
from datasets import Dataset
from datasets import DatasetDict


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
                 input_columns: Union[List[str], str],
                 output_column: Union[List[str], str],
                 test_split: Optional[str] = 'test'
                 ) -> None:
        self.input_columns = input_columns
        self.output_column = output_column
        self.dataset = dataset
        if isinstance(self.dataset, DatasetDict):
            if test_split in self.dataset.keys():
                self.references = self.dataset[test_split][self.output_column]
        elif isinstance(self.dataset, Dataset):
            self.references = self.dataset[self.output_column]
