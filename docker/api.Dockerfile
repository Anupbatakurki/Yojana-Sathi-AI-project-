# ---------- Yojana Saathi API ----------
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/api.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY pyproject.toml README.md /src/
COPY saathi /src/saathi
COPY data/schemes.json /src/data/schemes.json
RUN pip install --no-deps /src
# Bake the embedding model + vector index into the image -> no downloads and no cold start at runtime.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV CACHE_DIR=/opt/cache SCHEMES_PATH=/src/data/schemes.json EMBED_MODEL=${EMBED_MODEL} LOG_JSON=false
RUN mkdir -p /opt/cache && (python -c "from saathi.pipeline import SaathiPipeline; SaathiPipeline()" \
    || echo "WARNING: model prefetch failed; it will download on first start")

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="yojana-saathi-api" \
      org.opencontainers.image.description="Multilingual RAG assistant for Indian welfare schemes" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/cache /app/.cache
COPY --chown=app:app data/schemes.json /app/data/schemes.json
RUN mkdir -p /app/var && chown app:app /app/var
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENV=prod PORT=8000 WEB_CONCURRENCY=1 EMBED_MODEL=${EMBED_MODEL} \
    CACHE_DIR=/app/.cache SCHEMES_PATH=/app/data/schemes.json FEEDBACK_PATH=/app/var/feedback.jsonl
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=4)"
CMD ["sh", "-c", "exec uvicorn saathi.api.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers --forwarded-allow-ips='*' --timeout-graceful-shutdown 20 --no-server-header"]
