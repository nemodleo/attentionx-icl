SHELL=/bin/bash

install:
	poetry install

run-sst2:
	poetry run python scripts/sst2.py

run-sst5:
	poetry run python scripts/sst5.py

run-ag_news:
	poetry run python scripts/ag_news.py

check-quality:
	poetry run flake8 iclx
	poetry run mypy iclx
