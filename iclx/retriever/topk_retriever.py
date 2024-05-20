import os
import gc
import tqdm
import copy
import faiss
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from loguru import logger
from accelerate import Accelerator

from iclx.retriever import BaseRetriever
from iclx.utils import DatasetReader
from iclx.utils import DatasetEncoder
from iclx.utils import DataCollatorWithPaddingAndCuda


class TopkRetriever(BaseRetriever):
    """Topk In-context Learning Retriever Class
        Class of Topk Retriever.

    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`.
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        gen_datalist = self.dataset_reader.generate_input_field_corpus(self.test_ds)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        encode_dataset = DatasetEncoder(gen_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        self.dataloader = DataLoader(encode_dataset, batch_size=self.batch_size, collate_fn=co)

        self.model = SentenceTransformer(sentence_transformers_model_name)

        self.model = self.model.to(self.device)
        self.model.eval()

        self.topk_index_path = kwargs.get('topk_index_path', None)
        self.res = None
        self.index = self.create_index()
        self.rtr_idx_list = self.knn_search(self._ice_num)
        self.clear_memory()

    def create_index(self):
        # Load index if it exists
        if self.topk_index_path is not None and os.path.exists(self.topk_index_path):
            logger.info(f"Loading topk index from file: {self.topk_index_path}")
            index = faiss.read_index(self.topk_index_path)
            if self.device == 'cuda':
                self.res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self.res, 0, index)
            return index
        if self.topk_index_path is not None and os.path.exists(self.topk_index_path) is False:
            logger.info(f"Index file not found: {self.topk_index_path}")
            logger.info("Creating index and saving to file...")

        # Create index
        self.select_datalist = self.dataset_reader.generate_input_field_corpus(self.index_ds)
        encode_datalist = DatasetEncoder(self.select_datalist, tokenizer=self.tokenizer)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.tokenizer, device=self.device)
        dataloader = DataLoader(encode_datalist, batch_size=self.batch_size, collate_fn=co)

        if self.device == 'cpu':
            logger.info("Creating faiss-cpu index")
            index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
        elif self.device == 'cuda':
            logger.info("Creating faiss-gpu index")
            self.res = faiss.StandardGpuResources()
            flat_config = faiss.GpuIndexFlatConfig()
            flat_config.device = 0
            dim = self.model.get_sentence_embedding_dimension()
            index = faiss.GpuIndexFlatIP(self.res, dim, flat_config)
        else:
            raise ValueError("Invalid device type. Please specify either 'cpu' or 'cuda'.")

        res_list = self.forward(dataloader, process_bar=True, information="Creating index for index set...")
        embed_list = np.stack([res['embed'] for res in res_list])
        index.add(embed_list) #!

        logger.info("Index created")

        # Save index to file
        if self.topk_index_path is not None and os.path.exists(self.topk_index_path) is False:
            logger.info(f"Saving index to file: {self.topk_index_path}")
            _index = index if self.device == 'cpu' else faiss.index_gpu_to_cpu(index)
            os.makedirs(os.path.dirname(self.topk_index_path), exist_ok=True)
            faiss.write_index(_index, self.topk_index_path)

        return index

    def clear_memory(self):
        # https://github.com/facebookresearch/faiss/issues/2752
        # https://github.com/facebookresearch/faiss/issues/2821
        self.index.reset()
        if self.res: self.res.noTempMemory()
        del self.index
        del self.model
        del self.tokenizer
        del self.dataloader
        if self.res: del self.res
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def knn_search(self, ice_num):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        if ice_num == 0:
            return rtr_idx_list
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, ice_num)[1][0].tolist()
            rtr_idx_list[idx] = near_ids #!
        return rtr_idx_list

    def forward(self, dataloader, process_bar=False, information=''):
        res_list = []
        _dataloader = copy.deepcopy(dataloader)
        if process_bar:
            logger.info(information)
            _dataloader = tqdm.tqdm(_dataloader, disable=not self.is_main_process)
        for _, entry in enumerate(_dataloader):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                raw_text = self.tokenizer.batch_decode(entry['input_ids'], skip_special_tokens=True, verbose=False)
                res = self.model.encode(raw_text, show_progress_bar=False)
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def retrieve(self):
        return [rtr_idx[:self._ice_num] for rtr_idx in self.rtr_idx_list] #!
