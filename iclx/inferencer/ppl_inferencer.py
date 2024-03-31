import json
import numpy as np
import torch
from typing import List
from typing import Optional
from tqdm import trange
from loguru import logger
from accelerate import Accelerator

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
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, batch_size, accelerator,
                         output_json_filepath, output_json_filename, **kwargs)
        self.labels = labels

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
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
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template, pseudo_gt=pseudo_gt))
        output_handler.save_ice(ice)

        # 5. Calculating PPL for prompts in each label's class
        for label in labels:
            index = 0
            prompt_list = []
            sub_ppl_list = []

            # 5.1 Generate prompts of current label and truncate
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template, remain_sep=None)
                if self.max_model_token_num is not None:
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                        prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template)
                        prompt_token_num = self.get_input_token_num(prompt)

                prompt_list.append(prompt)

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                with torch.no_grad():
                    sub_res = self.__get_ppl(sub_prompt_list).tolist()
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
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

    def __get_ppl(self, input_texts: List[str], mask_length=None):
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
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
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
