FROM ubuntu:22.04

WORKDIR /CILO
COPY . /CILO

RUN apt update && apt upgrade -y

# Install dependencies
RUN apt install -y \
    patchelf \
    libglew-dev \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libopengl0 \
    ninja-build \
    build-essential \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN mkdir -p /root/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /miniconda3/miniconda.sh && \
    /root/miniconda3/bin/conda init bash

# Install Conda environment
RUN /root/miniconda3/bin/conda env create -f /CILO/dependencies/environment.yml && \
    rm -rf /root/.cache/pip

# Install dependencies for signatory and gym
RUN /root/miniconda3/bin/conda run -n continuous pip install setuptools==59.5.0 importlib-metadata==4.13.0

# Install other dependencies
RUN /root/miniconda3/bin/conda run -n continuous pip install -r /CILO/dependencies/requirements.txt

# Install torch and Signatory
RUN /root/miniconda3/bin/conda run -n continuous pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 
RUN /root/miniconda3/bin/conda run -n continuous pip install signatory==1.2.6.1.9.0

# Activate Conda environment and set it as default
SHELL ["/root/miniconda3/bin/conda", "run", "-n", "continuous", "/bin/bash", "-c"]
RUN echo "conda activate continuous" >> ~/.bashrc
