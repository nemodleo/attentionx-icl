import os
import torch
from typing import Any
from typing import List
from typing import Optional
from typing import Union
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate import init_empty_weights
from accelerate import infer_auto_device_map

from iclx.retriever.base_retriever import BaseRetriever
from iclx.utils.prompt_template import PromptTemplate


class BaseInferencer:
    """Basic In-context Learning Inferencer Class
        Base class of In-context Learning Inferencer, with no inference method.

    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class.
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file.
    """

    def __init__(self,
                 model_name: Optional[Union[str, Any]] = 'gpt2-xl',
                 tokenizer_name: Optional[Union[str, Any]] = None,
                 max_model_token_num: Optional[int] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./iclx_output",
                 output_json_filename: Optional[str] = "predictions",
                 task_description: str = None,
                 **kwargs
                 ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.accelerator = accelerator
        self.is_main_process = True if self.accelerator is None or self.accelerator.is_main_process else False
        self.task_description = task_description

        self._init_model(self.model_name)
        self._init_tokenizer(self.tokenizer_name)

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device not in self.model.device.type:
            raise ValueError(f"Model is not on the same device as the current device, device={self.device} != model.device={self.model.device.type}")
        self.model.eval()

        self.max_model_token_num = max_model_token_num or self.tokenizer.model_max_length
        if not self.max_model_token_num:
            return ValueError("tokenizer.model_max_length is not defined, please provide max_model_token_num")
        if self.tokenizer.model_max_length:
            if self.tokenizer.model_max_length < self.max_model_token_num:
                raise ValueError(f"max_model_token_num should be less than or equal to tokenizer.model_max_length, but got {self.max_model_token_num}")
        if self.max_model_token_num < 0:
            raise ValueError(f"max_model_token_num should be greater than or equal to 0, but got {self.max_model_token_num}")

        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        if not os.path.exists(self.output_json_filepath):
            os.makedirs(self.output_json_filepath)

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  use_ordering: Optional[bool] = False,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None
                  ) -> List:
        """Perform In-Context Inference given a retriever and optional templates.

        Args:
            retriever (:obj:`BaseRetriever`): An instance of a Retriever class that will be used to retrieve in-context examples
            ice_template (:obj:`PromptTemplate`, optional): A template for generating the in-context examples prompt. Defaults to None.
            prompt_template (:obj:`PromptTemplate`, optional): A template for generating the final prompt. Defaults to None.
            use_ordering (:obj:`bool`, optional): A flag to indicate whether to use ordering. Defaults to False.
            output_json_filepath (:obj:`str`, optional): The file path to save the results as a `JSON` file. Defaults to None.
            output_json_filename (:obj:`str`, optional): The file name to save the results as a `JSON` file. Defaults to None.

        Raises:
            NotImplementedError: If the function is not implemented in the subclass.

        Returns:
            :obj:`List:` A list of string, each representing the results of one inference.
        """
        raise NotImplementedError("Method hasn't been implemented yet")

    def _init_model(self, model_name):
        # model_config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          device_map="auto",
                                                          torch_dtype=torch.float16)
        # with init_empty_weights():
        #     empty_model = AutoModelForCausalLM.from_config(model_config)
        # device_map = infer_auto_device_map(empty_model, dtype="float16")
        # self.model = AutoModelForCausalLM.from_pretrained(model_name,
        #                                                   device_map=device_map,
        #                                                   offload_folder="offload",
        #                                                   offload_state_dict=True,
        #                                                   torch_dtype=torch.float16)

    def _init_tokenizer(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _add_task_description(self, separator, prompt):
        if not self.task_description:
            return prompt
        else:
            return self.task_description + separator + prompt

    def get_input_token_num(self, inputs):
        return len(self.tokenizer(inputs, verbose=False)['input_ids'])
