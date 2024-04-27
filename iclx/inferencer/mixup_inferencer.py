import copy
import json
import numpy as np
import torch
from typing import List
from typing import Dict
from typing import Optional
from tqdm import trange
from loguru import logger
from accelerate import Accelerator

from iclx.inferencer import BaseInferencer
from iclx.retriever import BaseRetriever
from iclx.utils import MixupPromptTemplate


class MixupInferencer(BaseInferencer):
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
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, batch_size, accelerator,
                         output_json_filepath, output_json_filename, task_description, **kwargs)
        self.labels = labels

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[MixupPromptTemplate] = None,
                  prompt_template: Optional[MixupPromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None,
                  pseudo_gt: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = MixupInferencerOutputHandler(self.accelerator)

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
            inputs_list = []
            prompt_list = []
            sub_ppl_list = []

            # 5.1 Generate prompts of current label and truncate
            for idx in range(len(ice_idx_list)):
                prompt = self._add_task_description(retriever.ice_separator, "")
                inputs = self.tokenizer(prompt, padding=True, return_tensors='pt', truncation=True)
                inputs = self.__convert_input_ids_to_embeds(inputs)
                postfix = retriever.generate_label_prompt(idx, "", label, ice_template=ice_template, prompt_template=prompt_template)
                postfix_inputs = self.tokenizer(postfix, padding=True, return_tensors='pt', truncation=True)
                postfix_inputs = self.__convert_input_ids_to_embeds(postfix_inputs)
                postfix_length = postfix_inputs["inputs_embeds"].shape[0]
                for ice_idx in ice_idx_list[idx]:
                    item = retriever.index_ds[ice_idx]
                    label = item[retriever.dataset_reader.output_column]
                    tp, labels, probs = ice_template.generate_ice_item(item)
                    mixup_inputs = self.__get_mixup_results(tp, labels, probs)
                    mixup_length = mixup_inputs["inputs_embeds"].shape[0]

                    if self.max_model_token_num is not None \
                        and inputs["inputs_embeds"].shape[0] + mixup_length + postfix_length > self.max_model_token_num:
                        break
                    else:
                        inputs = self.__merge_processed_results(inputs, mixup_inputs)
                        prompt += tp + " " + label

                inputs_list.append(self.__merge_processed_results(inputs, postfix_inputs))
                prompt_list.append(prompt + postfix)

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            self.batch_size = 1
            for idx in trange(0, len(inputs_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                inputs = inputs_list[idx]
                with torch.no_grad():
                    sub_res = self.__get_ppl(inputs).tolist()
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

    def __merge_processed_results(self, res1: Dict, res2: Dict):
        merged = dict()
        for k, v in merged.items():
            merged[k] = torch.cat((res1, res2), -1)
        return merged

    def __convert_input_ids_to_embeds(self, inputs: Dict):
        # TODO: remove side effect
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        embed_layer = self.model.get_input_embeddings()
        input_ids = inputs["input_ids"]
        inputs_embeds = embed_layer(input_ids)
        inputs["inputs_embeds"] = inputs_embeds
        return inputs

    def __get_mixup_results(self,
                            ice_text: str,
                            answer_tokens: List[str],
                            probs: List[float]):
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(ice_text, padding=True, return_tensors='pt', truncation=True)

        answer_embed = None
        for answer_token, prob in zip(answer_tokens, probs):
            answer_token_inputs = self.tokenizer(" " + answer_token, padding=True, return_tensors='pt', truncation=True)
            answer_token_inputs = self.__convert_input_ids_to_embeds(answer_token_inputs)

            if answer_embed:
                answer_embed += prob * answer_token_inputs["inputs_embeds"]
            else:
                answer_embed = prob * answer_token_inputs["inputs_embeds"]

        prob_sum = np.sum(probs)
        answer_embed /= prob_sum

        answer_token_inputs["input_embeds"] = answer_embed
        inputs = self.__merge_processed_results(inputs, answer_token_inputs)

        return inputs

    def __get_ppl(self, inputs: torch.tensor, mask_length=None):
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()

        # Prepare inputs
        inputs.pop("input_ids")
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss


class MixupInferencerOutputHandler:
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
