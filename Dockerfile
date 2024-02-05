FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update -y && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    libgl1 \
    build-essential \
    cmake \
    gcc \
    unzip \
    libpq-dev \
    libsndfile1-dev
RUN git lfs install

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN autotrain setup

COPY train.sh .
CMD ["./train.sh"]