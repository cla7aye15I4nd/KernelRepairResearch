FROM ubuntu:24.04

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
    wget && \
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
    python3 -m venv /opt/venv && \
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

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gnupg && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    nvidia-container-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN pip install \
    bs4 \
    torch \
    torchaudio \
    torchvision \
    unidiff
