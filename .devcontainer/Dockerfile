FROM python:3.14-slim-bookworm AS runtime-base

ENV DEBIAN_FRONTEND=noninteractive

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    less \
    openssh-client \
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

FROM runtime-base AS development

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt

RUN python3 -m pip install \
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

ENV PYTHONPATH=/workspace/src

FROM development AS release-builder

USER root

COPY pyproject.toml README.md /build/
COPY src /build/src

RUN python3 -m venv /opt/railbound-venv \
    && /opt/railbound-venv/bin/python -m pip install --no-cache-dir /build

FROM runtime-base AS release

COPY --from=release-builder /opt/railbound-venv /opt/railbound-venv

ENV PATH=/opt/railbound-venv/bin:$PATH

RUN mkdir -p /workspace \
    && chown vscode:vscode /workspace

USER vscode

WORKDIR /workspace

ENTRYPOINT ["railbound-grid"]
CMD ["--help"]
