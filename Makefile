$(eval export PATH=$(HOME)/.local/bin:$(PATH))
ifneq (, $(shell which poetry))
	AUTO_POETRY = poetry run
endif

poetry-install:
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install

poetry-faiss-gpu-reinstall:
	$(MAKE) poetry-remove-faiss-gpu
	poetry run $(MAKE) poetry-install-onemkl
	poetry run $(MAKE) poetry-build-faiss-gpu

poetry-remove-faiss-gpu:
	poetry remove faiss-gpu

poetry-install-onemkl:
	curl https://registrationcenter-download.intel.com/akdlm/IRC_NAS/adb8a02c-4ee7-4882-97d6-a524150da358/l_onemkl_p_2023.2.0.49497.sh --output onemkl.sh \
		&& bash onemkl.sh -a -s --eula accept \
		&& rm onemkl.sh

poetry-build-faiss-gpu:
	git clone --depth 1 --branch v1.8.0 https://github.com/facebookresearch/faiss.git \
		&& cd faiss \
		&& . /opt/intel/oneapi/setvars.sh \
		&& cmake -B build . \
		-DFAISS_ENABLE_GPU=ON \
		-DFAISS_ENABLE_PYTHON=ON \
		-DFAISS_ENABLE_RAFT=OFF \
		-DBUILD_TESTING=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DFAISS_ENABLE_C_API=OFF \
		-DCMAKE_BUILD_TYPE=Release \
		-DFAISS_OPT_LEVEL=avx2 \
		-DBLA_VENDOR=Intel10_64lp \
		"-DMKL_LIBRARIES=-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_gnu_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl" \
		-DCUDAToolkit_ROOT=/usr/local/cuda \
		-DCMAKE_CUDA_ARCHITECTURES="80;75" \
		-DPython_EXECUTABLE=$(shell which python) \
		&& make -C build -j$(nproc) faiss \
		&& make -C build -j$(nproc) swigfaiss \
		&& cd build/faiss/python && python setup.py install

poetry-export:
	poetry export -f requirements.txt --without-hashes --output requirements.txt

train-bert:
	$(AUTO_POETRY) python scripts/train_bert.py --dataset=$(DATASET) --lr=$(LR) --batch_size=$(BATCH_SIZE) --sampling_rate=$(SAMPLING_RATE) --n_gpus=$(N_GPUS)

infer-bert:
	$(AUTO_POETRY) python scripts/infer_bert.py --checkpoint_path=$(CHECKPOINT_PATH) --dataset=$(DATASET) --dataset_split=$(PHASE) --batch_size=$(BATCH_SIZE) --file_name=$(FILE_NAME) --sampling_rate=$(SAMPLING_RATE)

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
