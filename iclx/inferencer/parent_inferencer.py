# create soft labels from larger model
# dilstill GPT2 : 82M params
import torch
from typing import Any
from typing import List
from typing import Optional
from typing import Union
from tqdm import trange
from loguru import logger
from accelerate import Accelerator

from iclx.utils.prompt_template import PromptTemplate
from iclx.inferencer.ppl_inferencer import PPLInferencer
from iclx.inferencer.ppl_inferencer import PPLInferencerOutputHandler
from iclx.retriever import BaseRetriever


class ParentInferencer(PPLInferencer):

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[Union[str, Any]] = None,
                 max_model_token_num: Optional[int] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./iclx_output",
                 output_json_filename: Optional[str] = "predictions",
                 labels: Optional[List] = None,
                 task_description: str = None,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, batch_size, accelerator,
                         output_json_filepath, output_json_filename, task_description, **kwargs)
        self.labels = labels

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None,
                  normalizing_str: Optional[str] = None
                  ) -> List:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)

        sub_predictions = []
        ppl = []
        ice = []

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Get labels of all the classes
        if self.labels is None:
            labels = retriever.get_labels(ice_template=ice_template, prompt_template=prompt_template)
        else:
            labels = self.labels

        # 4. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template))
        output_handler.save_ice(ice)

        # 5. Calculating PPL for prompts in each label's class
        for label in labels:
            index = 0
            prompt_list = []
            sub_ppl_list = []
            normalizing_prompt_list = []
            context_length_list = []

            # 5.1 Generate prompts of current label and truncate
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                         prompt_template=prompt_template,
                                                         remain_sep=normalizing_str is not None)
                prompt = self._add_task_description(retriever.ice_separator, prompt)
                if self.max_model_token_num is not None and self.api_name != 'gpt3':
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                        prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                                 prompt_template=prompt_template)
                        prompt = self._add_task_description(retriever.ice_separator, prompt)
                        prompt_token_num = self.get_input_token_num(prompt)

                if normalizing_str is not None:
                    prompt_sep = prompt
                    if prompt_template is not None:
                        sep_token = prompt_template.sep_token
                    else:
                        sep_token = ice_template.sep_token
                    sep_pos = prompt_sep.find(sep_token)

                    context = prompt_sep[0:sep_pos]
                    answer = prompt_sep[sep_pos:].replace(sep_token, '')
                    prompt = context + answer
                    normalizing_prompt = normalizing_str + answer

                    context_length_list.append(self.get_input_token_num(context))
                    normalizing_prompt_list.append(normalizing_prompt)
                prompt_list.append(prompt)

            if normalizing_str is not None:
                normalizing_str_len = self.get_input_token_num(normalizing_str)

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                if normalizing_str is not None:
                    sub_context_length_list = context_length_list[idx:idx + self.batch_size]
                    sub_normalizing_prompt_list = normalizing_prompt_list[idx:idx + self.batch_size]

                with torch.no_grad():
                    if normalizing_str is not None:
                        res1 = self.__get_ppl(input_texts=sub_prompt_list, mask_length=sub_context_length_list)
                        res2 = self.__get_ppl(input_texts=sub_normalizing_prompt_list,
                                              mask_length=[normalizing_str_len for i in range(len(sub_prompt_list))]
                                              )
                        sub_res = res1 - res2
                    else:
                        sub_res = self.__get_ppl(sub_prompt_list).tolist()
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. DO NOT lowest PPL class as predictions. rather, save them all.
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append({idx: ppl for idx, ppl in enumerate(single_ppl)})

        return sub_predictions
