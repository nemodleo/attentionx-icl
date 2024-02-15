# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes
ARG PYTORCH_VERSION=24.01
FROM nvcr.io/nvidia/pytorch:${PYTORCH_VERSION}-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y sudo tree python3-pip
RUN python -m pip install --upgrade pip

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh
RUN apt-get install -y git-lfs
RUN curl https://getcroc.schollz.com | bash
    
RUN mkdir /app
ARG requirements
RUN echo "$requirements"
RUN echo "$requirements" > /app/requirements.txt
RUN pip install -r /app/requirements.txt

RUN mkdir /work
WORKDIR /work

RUN python --version && \
    pip --version && \
    pip list | grep torch && \
    python -c "import torch ; print(torch.__version__)"


