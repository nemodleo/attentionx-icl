$(eval export PATH=$(HOME)/.local/bin:$(PATH))
ifneq (, $(shell which poetry))
	AUTO_POETRY = poetry run
endif

poetry-install:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

poetry-export:
	poetry export -f requirements.txt --without-hashes --output requirements.txt

train-bert:
	$(AUTO_POETRY) python scripts/train_bert.py $(DATASET)

infer-bert:
	$(AUTO_POETRY) python scripts/infer_bert.py $(CHECKPOINT_PATH) $(DATASET) $(BATCH_SIZE) $(FILE_NAME)

run-sst2:
	$(AUTO_POETRY) python scripts/sst2.py

run-create_train:
	$(AUTO_POETRY) python scripts/create_train_with_pseudo.py $(SETUP_DICT)

run-sst2_gpt_neo_2.7B:
	$(AUTO_POETRY) python scripts/sst2_gpt_neo_2.7B.py

run-sst2_topk:
	$(AUTO_POETRY) python scripts/sst2_topk.py

run-sst5:
	$(AUTO_POETRY) python scripts/sst5.py

run-ag_news:
	$(AUTO_POETRY) python scripts/ag_news.py

run-create_data_sst2:
	$(AUTO_POETRY) python scripts/create_traindata_sst2.py

run-create_data_sst5:
	$(AUTO_POETRY) python scripts/create_traindata_sst5.py

run-create_subj:
	$(AUTO_POETRY) python scripts/bert/create_traindata_subj.py

run-trec:
	$(AUTO_POETRY) python scripts/trec.py

vessl-workspace-init:
	mkdir /root/.cache
	ln -s /opt/.cache/huggingface /root/.cache/huggingface

run-create_train:
	$(AUTO_POETRY) python scripts/create_train_with_pseudo.py $(SETUP_DICT)

run-distill:
	$(AUTO_POETRY) python scripts/distill.py $(SETUP_DICT)

check-quality:
	$(AUTO_POETRY) flake8 --ignore=E501 iclx data_utils scripts
	$(AUTO_POETRY) mypy iclx data_utils scripts

docker-build:
	# mv ~/.cache/huggingface/ huggingface/
	docker build -t nemodleosnu/iclx:0.2.1 -f Dockerfile .
	# mv huggingface/ ~/.cache/huggingface/

docker-push:
	docker push nemodleosnu/iclx:0.2.1

docker-build-and-push: 
	$(MAKE) docker-build 
	$(MAKE) docker-push

mac-docker-build:
	# mv ~/.cache/huggingface/ huggingface/
	docker build --platform linux/amd64 -t nemodleosnu/iclx:0.2.1 -f Dockerfile .
	# mv huggingface/ ~/.cache/huggingface/

mac-docker-build-and-push: 
	$(MAKE) mac-docker-build 
	$(MAKE) docker-push

download-dataset:
	vessl dataset download iclx / ${INPUT}

do-symlink:
	ln -s ${INPUT}/ckpt ckpt
	ln -s ${INPUT}/data data
