FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

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
    zip \
    libpq-dev \
    libsndfile1-dev
RUN git lfs install

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN git clone -b main https://github.com/huggingface/diffusers.git

WORKDIR /app/diffusers
RUN pip install -e .

WORKDIR /app/diffusers/examples/dreambooth
RUN pip install -r requirements_sdxl.txt
RUN pip install -r requirements.txt
RUN accelerate config default

ENV HF_HOME /models

COPY train.py .
CMD ["python", "train.py"]