SHELL=/bin/bash

install:
	IS_POETRY_INSTALLED="$(which poetry)"
	if [ -z "$IS_POETRY_INSTALLED" ]; then
		curl -sSL https://install.python-poetry.org | python3 -
		export PATH="$HOME/.local/bin:$PATH"
	fi
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
