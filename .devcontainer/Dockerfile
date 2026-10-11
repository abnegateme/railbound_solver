FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    less \
    openssh-client \
    python3 \
    python3-dev \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd vscode \
    && useradd \
    --gid vscode \
    --create-home \
    --shell /bin/bash \
    vscode \
    && echo "vscode ALL=(root) NOPASSWD:ALL" \
    > /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

COPY .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt

RUN python3 -m pip install \
    --break-system-packages \
    --no-cache-dir \
    --retries 8 \
    -r /tmp/requirements-dev.txt \
    && rm /tmp/requirements-dev.txt

RUN mkdir -p /workspace /home/vscode/.codex \
    && chown -R vscode:vscode \
    /workspace \
    /home/vscode

USER vscode

WORKDIR /workspace
