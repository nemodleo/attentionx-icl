# attentionx-icl

exp environments for icl

this repository is a modification of https://github.com/Shark-NLP/OpenICL

[vessl-run env guide](https://www.notion.so/minchan0502/vessl-run-env-guide-620e400e19754fcdb6819773f818318c)

## Env

1. using poetry 
```bash
make poetry-install
```

2. using docker
https://github.com/nemodleo/attentionx-icl/issues/19
```bash
pip install transformers==4.39.3
```

3. symlinking for using cache ```make vessl-workspace-init```


## Ex

```
make run-sst2
make run-sst2_gpt_j_6B
make run-sst2_gpt_neo_2.7B
make run-sst2_topk
make run-sst5
make run-ag_news
make run-trec
```

```
make vessl-run-sst2
make vessl-run-sst2_gpt_j_6B
make vessl-run-sst2_gpt_neo_2.7B
make vessl-run-sst2_topk
make vessl-run-sst5
make vessl-run-ag_news
make vessl-run-trec
```


## TODO

[wip] implementation iclx more feature

[wip] more examples
