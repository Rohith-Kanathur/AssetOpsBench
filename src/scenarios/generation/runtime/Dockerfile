FROM node:22-bookworm-slim AS node
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @openai/codex@0.154.0
COPY requirements.txt /opt/generation/requirements.txt
RUN uv pip install --system -r /opt/generation/requirements.txt
ENV PYTHONPATH=/workspace/src PYTHONUNBUFFERED=1
WORKDIR /workspace
