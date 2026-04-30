ARG PYTHON_IMAGE=python:3.12.7-slim-bookworm
FROM ${PYTHON_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CHERAGH_INDEX=/data/.cheragh_index \
    CHERAGH_INDEX_ROOT=/data \
    CHERAGH_ENABLE_INDEXING=false

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app
COPY pyproject.toml README.md LICENSE /app/
COPY docker/constraints.txt /app/docker/constraints.txt
COPY src /app/src
COPY docs /app/docs
COPY examples /app/examples
RUN python -m pip install --upgrade "pip==24.3.1" \
    && pip install --constraint /app/docker/constraints.txt ".[fastapi,config,qdrant]" \
    && mkdir -p /data \
    && chown -R app:app /app /data

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["cheragh", "serve", "--index", "/data/.cheragh_index", "--host", "0.0.0.0", "--port", "8000", "--index-root", "/data"]
