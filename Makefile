SHELL=/bin/bash

install:
	poetry install

run-test:
	poetry run python scripts/test.py

check-quality:
	poetry run flake8 iclx
	poetry run mypy iclx
