ifneq (, $(shell which poetry))
	AUTO_POETRY = poetry run
endif

poetry-install:
	curl -sSL https://install.python-poetry.org | python3 -
	export PATH="${HOME}/.local/bin:${PATH}"
	poetry install

poetry-export:
	poetry export -f requirements.txt > requirements.txt

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
	docker build -t nemodleosnu/iclx:0.1.1 --build-arg requirements="$(cat requirements.txt)" -f Dockerfile .

docker-build-and-push-vessl:
	apt-get -qq install docker.io
	set -x
	dockerd -b none --iptables=0 -l warn &
	for i in $(seq 5); do [ ! -S "/var/run/docker.sock" ] && sleep 2 || break; done
	docker build -t nemodleosnu/iclx:0.1.1 --build-arg requirements="$(cat requirements.txt)" -f Dockerfile .
	docker push nemodleosnu/iclx:0.1.1
	kill $(jobs -p)

docker-push:
	docker push nemodleosnu/iclx:0.1.1
