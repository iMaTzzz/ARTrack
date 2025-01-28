FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    gcc \
    g++ \
    bash \
    vim \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Clone ARTrack repository and checkout ARTrackV2 branch
RUN git clone https://github.com/iMaTzzz/ARTrack.git /workspace/ARTrack && \
    cd /workspace/ARTrack && \
    git checkout ARTrackV2

# Copy required files
COPY hand.mp4 /workspace/ARTrack.
COPY artrackv2_seq_256_full.pth.tar /workspace/ARTrack.

# Set working directory
WORKDIR /workspace/ARTrack

# Create the conda environment
RUN conda env create -f /workspace/ARTrack/ARTrack_env_cuda113.yaml

# Set the default command
CMD ["/bin/bash"]