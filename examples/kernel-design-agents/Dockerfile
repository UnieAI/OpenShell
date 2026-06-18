FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ARG CODEX_NPM_PACKAGE="@openai/codex"
ARG CODEX_VERSION="0.130.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    iproute2 \
    jq \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000660000 sandbox \
    && useradd -m -u 1000660000 -g 1000660000 -s /bin/bash sandbox \
    && mkdir -p /app /sandbox \
    && touch /home/sandbox/.profile /home/sandbox/.bashrc \
    && chown -R sandbox:sandbox /home/sandbox /app /sandbox \
    && chmod 755 /home/sandbox \
    && chmod 644 /home/sandbox/.profile /home/sandbox/.bashrc

WORKDIR /app

COPY . /app

RUN npm install -g "${CODEX_NPM_PACKAGE}@${CODEX_VERSION}" \
    && codex --version \
    && chmod +x /app/*.sh /app/scripts/*.sh \
    && chown -R sandbox:sandbox /app

ENV KDA_WORKSPACE_ROOT=/sandbox
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
