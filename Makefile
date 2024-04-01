$(eval export PATH=$(HOME)/.local/bin:$(PATH))
ifneq (, $(shell which poetry))
	AUTO_POETRY = poetry run
endif

poetry-install:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

poetry-export:
	poetry export -f requirements.txt --without-hashes > requirements.txt

run-sst2:
	$(AUTO_POETRY) python scripts/sst2.py

run-sst5:
	$(AUTO_POETRY) python scripts/sst5.py

run-ag_news:
	$(AUTO_POETRY) python scripts/ag_news.py

run-create_data_sst2:
	$(AUTO_POETRY) python scripts/create_traindata_sst2.py

run-create_subj:
	$(AUTO_POETRY) python scripts/create_subj_train.py

run-create_train:
	$(AUTO_POETRY) python scripts/create_train_with_pseudo.py $(SETUP_DICT)

run-trec:
	$(AUTO_POETRY) python scripts/trec.py

run-distill:
	$(AUTO_POETRY) python scripts/distill.py $(SETUP_DICT)

check-quality:
	$(AUTO_POETRY) flake8 iclx
	$(AUTO_POETRY) mypy iclx

docker-build:
	docker build -t nemodleosnu/iclx:0.1.3 -f Dockerfile .

docker-push:
	docker push nemodleosnu/iclx:0.1.3
