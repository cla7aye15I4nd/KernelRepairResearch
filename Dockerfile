FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    bear \
    build-essential \
    clang \
    clang-16 \
    clangd \
    cmake \
    curl \
    gdb \
    git \
    libclang-rt-dev \
    lld \
    llvm \
    make \
    pkg-config \
    python3 \
    python3-pip \
    python3.12-venv \
    ssh \
    sudo \
    tmux \
    wget \
    ca-certificates \
    gnupg && \
    # Install Docker
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin && \
    # Create Python virtual environment
    python3 -m venv /opt/venv && \
    # Clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

RUN cd /tmp && \
    wget https://github.com/universal-ctags/ctags/releases/download/v6.2.0/universal-ctags-6.2.0.tar.gz && \
    tar -xzf universal-ctags-6.2.0.tar.gz && \
    cd universal-ctags-6.2.0 && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    cd / && \
    rm -rf /tmp/universal-ctags-6.2.0*

RUN pip install --upgrade pip && \
    pip install \
    bs4 \
    numpy \
    protobuf \
    scikit-learn \
    tqdm \
    transformers \
    unidiff && \
    pip install torch --index-url https://download.pytorch.org/whl/cu129 && \
    pip install torchaudio --index-url https://download.pytorch.org/whl/cu129 && \
    pip install torchvision --index-url https://download.pytorch.org/whl/cu129
