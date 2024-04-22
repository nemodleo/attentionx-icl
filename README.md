# ICKD: In-Context Knowledge Distillation

*This repository is built upon Shark-NLP's [OpenICL]( https://github.com/Shark-NLP/OpenICL) framework*

## Environment Setup
Refer to [vessl-run env guide](https://www.notion.so/minchan0502/vessl-run-env-guide-620e400e19754fcdb6819773f818318c) for detailed explanation on how to setup environment in vessl
1. using poetry 
```bash
make poetry-install
```

2. using docker
```bash
```

3. symlinking for using cache ```ln -s /opt/.cache/huggingface /root/.cache/huggingface```


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