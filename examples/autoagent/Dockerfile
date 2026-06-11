FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel

ARG CODEX_NPM_PACKAGE="@openai/codex"
ARG CODEX_VERSION="0.130.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    nodejs \
    npm \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g 1000660000 sandbox \
    && useradd -m -u 1000660000 -g 1000660000 -s /bin/bash sandbox \
    && mkdir -p /app /sandbox \
    && touch /home/sandbox/.profile /home/sandbox/.bashrc \
    && chown -R sandbox:sandbox /home/sandbox /app /sandbox \
    && chmod 755 /home/sandbox \
    && chmod 644 /home/sandbox/.profile /home/sandbox/.bashrc

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
RUN python3 -m venv --system-site-packages /app/.venv \
    && /app/.venv/bin/python -m pip install --upgrade pip \
    && /app/.venv/bin/python -m pip install --no-cache-dir \
        apache-tvm-ffi \
        Jinja2 \
        optuna \
        PyYAML \
    && npm install -g "${CODEX_NPM_PACKAGE}@${CODEX_VERSION}" \
    && ln -sf /app/.venv/bin/python /usr/local/bin/python \
    && ln -sf /app/.venv/bin/python /usr/local/bin/python3 \
    && ln -sf /app/.venv/bin/pip /usr/local/bin/pip \
    && ln -sf /app/.venv/bin/pip /usr/local/bin/pip3 \
    && /app/.venv/bin/python -c "import sys, torch, tvm_ffi; print(sys.executable); print(torch.__version__); print(tvm_ffi.__file__)" \
    && codex --version

COPY . /app
RUN chown -R sandbox:sandbox /app

ENV AUTOAGENT_WORKSPACE=/app
ENV AUTOAGENT_STATE_DIR=/sandbox/autoagent
ENV AUTOAGENT_PROFILE_EXECUTOR=disabled
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/app/.venv/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
