FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Copy GIT repo
COPY . /workspace/
WORKDIR /workspace

# Create the conda environment
RUN conda env create -f /workspace/ARTrack_env_cuda113.yaml

# Activate the conda environment
SHELL ["conda", "run", "-n", "artrack", "/bin/bash", "-c"]

# Set the default command
CMD ["conda", "run", "-n", "artrack", "/bin/bash"]
