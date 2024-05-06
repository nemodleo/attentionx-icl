# ICKD: In-Context Knowledge Distillation

*built using Shark-NLP's [OpenICL]( https://github.com/Shark-NLP/OpenICL) framework*

**Can we beat the ground truth using soft label during In-Context Learning?** This repository helps users to answer this question by allowing them to create soft labels on dataset of their own choice, and distill this soft label information to the student model during in-context learning.

-----

## Environment Setup
```bash
make poetry-install
```

if building faiss index is too slow in your gpu,
```bash
make poetry-faiss-gpu-reinstall
```


## Preparing Dataset (Generating Soft Label)
#### 1. Using a non-BERT teacher model
1. Download the `json` format data into an appropriate `data` folder
2. Create a config file under `config/data` following the [template](https://github.com/nemodleo/attentionx-icl/blob/develop/config/data/template_datagen-config.json) ([More information](https://www.notion.so/minchan0502/5795f433a8c74a728305be7937d0fb42?pvs=4) on template fields)
3. Run below command
   ```bash
   make run-create_train SETUP_DICT="config/data/<config_file_name>.json"
   ```
4. Use `data_utils/generated_train_dist.ipynb` to get dataset statistics

#### 2. Using BERT as a teacher model
Currently supports: `sst2, sst5, trec, ag_news, yelp, qnli, mnli`
```bash
# to pretrain BERT
make train-bert dataset="<dataset_name>"

# create train data using pretrained BERT
make infer-bert checkpoint_path="<path_to_ckpt>" dataset="<dataset_name>" file_name="<output_file_name>"
```

## Running Experiments (Knowledge Distillation to Student)
1. Create a config file under `config/distill` following the [template](https://github.com/nemodleo/attentionx-icl/blob/develop/config/distill/template_distill-config.json) ([More information](https://www.notion.so/minchan0502/5795f433a8c74a728305be7937d0fb42?pvs=4) on template fields)
2. Run below command
    ```bash
    make run-distill SETUP_DICT="config/distill/<config_file_name>.json"
    ```
3. `.txt` file with accuracies and `.png` file of corrresponding plots will be saved as artifacts