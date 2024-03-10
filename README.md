# attentionx-icl

exp environments for icl

this repository is a modification of https://github.com/Shark-NLP/OpenICL


## Env

1. using poetry 
```bash
make poetry-install
make do-symlink INPUT={cached ckpt-data dir}
```

2. using docker
```bash
make do-symlink INPUT=/input

```


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


## TODO

[wip] implementation iclx more feature

[wip] more examples
