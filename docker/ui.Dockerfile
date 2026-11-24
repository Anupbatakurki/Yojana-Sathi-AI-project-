# ---------- Yojana Saathi UI (Streamlit) ----------
# Talks to the API when API_URL is set; otherwise runs the pipeline in-process (single-container hosts).
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/ui.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY pyproject.toml README.md /src/
COPY saathi /src/saathi
COPY data/schemes.json /src/data/schemes.json
RUN pip install --no-deps /src
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV CACHE_DIR=/opt/cache SCHEMES_PATH=/src/data/schemes.json EMBED_MODEL=${EMBED_MODEL} LOG_JSON=false
RUN mkdir -p /opt/cache && (python -c "from saathi.pipeline import SaathiPipeline; SaathiPipeline()" \
    || echo "WARNING: model prefetch failed; it will download on first start")

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="yojana-saathi-ui" org.opencontainers.image.licenses="MIT"
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /opt/cache /app/.cache
COPY --chown=app:app data/schemes.json /app/data/schemes.json
COPY --chown=app:app streamlit_app.py /app/
COPY --chown=app:app .streamlit /app/.streamlit
RUN mkdir -p /app/var && chown app:app /app/var
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    ENV=prod PORT=8501 API_URL="" EMBED_MODEL=${EMBED_MODEL} \
    CACHE_DIR=/app/.cache SCHEMES_PATH=/app/data/schemes.json FEEDBACK_PATH=/app/var/feedback.jsonl
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health', timeout=4)"
CMD ["sh", "-c", "exec streamlit run streamlit_app.py --server.port ${PORT} --server.address 0.0.0.0"]
