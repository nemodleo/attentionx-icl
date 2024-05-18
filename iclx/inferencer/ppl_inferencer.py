import json
import copy
import numpy as np
import torch
from typing import List, Dict
from typing import Optional
from tqdm import tqdm
from tqdm import trange
from loguru import logger
from accelerate import Accelerator
from transformers.file_utils import ModelOutput

from iclx.inferencer import BaseInferencer
from iclx.retriever import BaseRetriever
from iclx.utils import PromptTemplate


class PPLInferencer(BaseInferencer):
    """PPL In-context Learning Inferencer Class
        Perplexity-based In-context Learning Inferencer.

    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class.
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file.
        labels (:obj:`List`, optional): A list of labels for all classes.
    """

    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./iclx_output",
                 output_json_filename: Optional[str] = "predictions",
                 labels: Optional[List] = None,
                 task_description: str = None,
                 use_cache: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, batch_size, accelerator,
                         output_json_filepath, output_json_filename, task_description, **kwargs)
        self.labels = labels
        self.use_cache = use_cache

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  use_ordering: Optional[bool] = False,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None,
                  pseudo_gt: Optional[str] = None) -> List:
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
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, pseudo_gt=pseudo_gt, use_ordering=use_ordering))
        output_handler.save_ice(ice)

        # 5. Calculating PPL for prompts in each label's class
        # self.use_cache = False
        if self.use_cache:
            logger.info("Using cache for PPL calculation")
            if self.batch_size != 1:
                raise ValueError("Batch size must be 1 when using cache for PPL calculation. Please set batch_size=1.")

            # 5.1 Generate prompts of current label and truncate
            _dummy_label = labels[0]
            prompt_wo_label_list = []
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt_wo_label_and_eos(idx, ice[idx], _dummy_label, ice_template=ice_template, prompt_template=prompt_template)
                prompt = self._add_task_description(retriever.ice_separator, prompt)
                if self.max_model_token_num is not None:
                    prompt_w_label = retriever.add_label_and_eos(prompt, _dummy_label, ice_template=ice_template, prompt_template=prompt_template)
                    prompt_token_num = self.get_input_token_num(prompt_w_label)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, use_ordering=use_ordering)
                        prompt = retriever.generate_label_prompt_wo_label_and_eos(idx, ice[idx], _dummy_label, ice_template=ice_template, prompt_template=prompt_template)
                        prompt = self._add_task_description(retriever.ice_separator, prompt)
                        prompt_w_label = retriever.add_label_and_eos(prompt, _dummy_label, ice_template=ice_template, prompt_template=prompt_template)
                        prompt_token_num = self.get_input_token_num(prompt_w_label)
                prompt_wo_label_list.append(prompt)

            # 5.2 Get PPL without label and eos token for recycle token
            logger.info(f"Calculating PPL for prompts labeled '{labels}'")

            sub_prompt_label_list_dict = {}
            sub_next_inputs_dict = {}
            for label in labels:
                prompt_label = retriever.get_label_and_eos(label, ice_template=ice_template, prompt_template=prompt_template)
                add_prompt_label = ' ' + prompt_label
                sub_prompt_label_list = [add_prompt_label] * self.batch_size
                sub_next_inputs = self._get_inputs(sub_prompt_label_list)
                sub_prompt_label_list_dict[label] = sub_prompt_label_list
                sub_next_inputs_dict[label] = sub_next_inputs


            index = 0
            n_labels = len(labels)
            for idx in trange(0, len(prompt_wo_label_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_wo_label_list = prompt_wo_label_list[idx:idx + self.batch_size]
                sub_inputs = self._get_inputs(sub_prompt_wo_label_list)
                with torch.no_grad():
                    sub_caches = self._get_cache(inputs=sub_inputs)

                sub_ppl_list = []
                for label in labels:
                    prompt_label = retriever.get_label_and_eos(label, ice_template=ice_template, prompt_template=prompt_template)
                    add_prompt_label = ' ' + prompt_label

                    # 5.3 Get PPL
                    sub_prompt_label_list = sub_prompt_label_list_dict[label]
                    sub_next_inputs = sub_next_inputs_dict[label]
                    with torch.no_grad():
                        sub_res = self._get_ppl_use_cache(sub_caches, inputs=sub_inputs, next_inputs=sub_next_inputs).tolist()
                    for res, prompt_wo_label, _add_prompt_label in zip(sub_res, sub_prompt_wo_label_list, sub_prompt_label_list):
                        sub_ppl_list.append(res)
                        prompt = prompt_wo_label + _add_prompt_label
                        output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index//n_labels)
                        index = index + 1
                ppl.append(sub_ppl_list)
        else:
            logger.info("Not using cache for PPL calculation")
            for label in labels:
                index = 0
                prompt_list = []
                sub_ppl_list = []

                # 5.1 Generate prompts of current label and truncate
                for idx in range(len(ice_idx_list)):
                    prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template, remain_sep=None)
                    prompt = self._add_task_description(retriever.ice_separator, prompt)
                    if self.max_model_token_num is not None:
                        prompt_token_num = self.get_input_token_num(prompt)
                        while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                            ice_idx_list[idx] = ice_idx_list[idx][:-1]
                            ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, use_ordering=use_ordering)
                            prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template)
                            prompt = self._add_task_description(retriever.ice_separator, prompt)
                            prompt_token_num = self.get_input_token_num(prompt)

                    prompt_list.append(prompt)

                # 5.2 Get PPL
                logger.info(f"Calculating PPL for prompts labeled '{label}'")
                for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                    sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                    with torch.no_grad():
                        sub_res = self._get_ppl(sub_prompt_list).tolist()
                    for res, prompt in zip(sub_res, sub_prompt_list):
                        sub_ppl_list.append(res)
                        output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
                        index = index + 1
                ppl.append(sub_ppl_list)
            ppl = list(zip(*ppl))
        # logger.info("ppl: {}".format(ppl)); exit()

        # 6. Get lowest PPL class as predictions
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # 7. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()]

    def _get_inputs(self,
                    input_texts: List[str]):
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs

    def _get_cache(self,
                   input_texts: Optional[List[str]]=None,
                   inputs: Optional[Dict[str, torch.Tensor]]=None):
        if input_texts is None and inputs is None:
            raise ValueError("Either input_texts or inputs must be provided.")
        _inputs = self._get_inputs(input_texts) if inputs is None else copy.deepcopy(inputs)
        outputs = self.model(**_inputs, use_cache=True)
        return outputs

    def _get_ppl_use_cache(self,
                           sub_caches: ModelOutput,
                           input_texts: Optional[List[str]]=None,
                           next_texts: Optional[List[str]]=None,
                           inputs: Optional[Dict[str, torch.Tensor]]=None,
                           next_inputs: Optional[Dict[str, torch.Tensor]]=None,
                           mask_length=None):
        if input_texts is None and inputs is None:
            raise ValueError("Either input_texts or inputs must be provided.")
        if next_texts is None and next_inputs is None:
            raise ValueError("Either next_texts or next_inputs must be provided.")

        _inputs = self._get_inputs(input_texts) if inputs is None else copy.deepcopy(inputs)
        _next_inputs = self._get_inputs(next_texts) if next_inputs is None else copy.deepcopy(next_inputs)
        _next_inputs.pop("attention_mask") #! only batch size 1

        outputs = self.model(**_next_inputs, past_key_values=sub_caches.past_key_values)

        logits = sub_caches.logits
        labels = _inputs["input_ids"]

        # remove pad token and logit
        mask_lengths = (labels == self.tokenizer.pad_token_id).int().argmax(-1)
        logits = [logit[:mask_length, :] if mask_length > 0 else logit for mask_length, logit in zip(mask_lengths, logits)]
        labels = [shift_label[:mask_length] if mask_length > 0 else shift_label for mask_length, shift_label in zip(mask_lengths, labels)]

        # add next token and logit
        logits = [torch.cat([logit, next_logit], dim=0) for logit, next_logit in zip(logits, outputs.logits)]
        labels = [torch.cat([label, next_label], dim=0) for label, next_label in zip(labels, _next_inputs["input_ids"])]

        # pad token and logit
        max_length = max([len(label) for label in labels])
        for i in range(len(labels)):
            pad_length = max_length - len(labels[i])
            if pad_length == 0:
                continue
            _dummy_logit = torch.zeros_like(logits[i][0])
            _pad_label = torch.full((pad_length,), fill_value=self.tokenizer.pad_token_id, device=labels[i].device)
            logits[i] = torch.cat([logits[i], _dummy_logit.unsqueeze(0).repeat(pad_length, 1)], dim=0)
            labels[i] = torch.cat([labels[i], _pad_label], dim=0)

        logits = torch.stack(logits)
        labels = torch.stack(labels)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (labels != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def _get_ppl(self, input_texts: List[str], mask_length=None):
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy() #!
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss


class PPLInferencerOutputHandler:
    results_dict = {}

    def __init__(self,
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        self.accelerator = accelerator
        self.results_dict = {}

    def subprocess_write_to_json(self, output_json_filepath: str, output_json_filename: str):
        if self.accelerator is not None:
            print(f'{output_json_filepath}/process{self.accelerator.process_index}_{output_json_filename}.json')
            with open(f'{output_json_filepath}/process{self.accelerator.process_index}_{output_json_filename}.json',
                      'w', encoding='utf-8') as json_file:
                json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
                json_file.close()

    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        with open(f'{output_json_filepath}/{output_json_filename}.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
            json_file.close()

    def merge_to_main_process(self, output_json_filepath: str, output_json_filename: str):
        if self.accelerator is not None and self.accelerator.is_main_process:
            for pid in range(self.accelerator.num_processes):
                with open(f'{output_json_filepath}/process{pid}_{output_json_filename}.json', 'r',
                          encoding='utf-8') as json_file:
                    subprocess_results_dict = json.load(json_file)
                    self.results_dict.update(subprocess_results_dict)
                    json_file.close()
            self.results_dict = dict(sorted(self.results_dict.items(), key=lambda x: int(x[0])))

    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example

    def save_predictions(self, predictions):
        for idx, prediction in enumerate(predictions):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['prediction'] = prediction

    def save_prompt_and_ppl(self, label, input, prompt, ppl, idx):
        if self.accelerator is not None:
            idx = idx * self.accelerator.num_processes + self.accelerator.process_index
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if 'label: ' + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]['label: ' + str(label)] = {}
        self.results_dict[str(idx)]['label: ' + str(label)]['testing input'] = input
        self.results_dict[str(idx)]['label: ' + str(label)]['prompt'] = prompt
        self.results_dict[str(idx)]['label: ' + str(label)]['PPL'] = ppl
