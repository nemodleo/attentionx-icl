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

run-trec:
	$(AUTO_POETRY) python scripts/trec.py

check-quality:
	$(AUTO_POETRY) flake8 iclx
	$(AUTO_POETRY) mypy iclx

docker-build:
	docker build -t nemodleosnu/iclx:0.1.2 -f Dockerfile .

docker-push:
	docker push nemodleosnu/iclx:0.1.2
