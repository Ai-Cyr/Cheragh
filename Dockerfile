ARG PYTHON_IMAGE=python:3.12.14-slim-bookworm

FROM ${PYTHON_IMAGE} AS builder

ARG PIP_VERSION=26.2.1
ARG BUILD_VERSION=1.5.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install \
        "pip==${PIP_VERSION}" \
        "build==${BUILD_VERSION}" \
        "setuptools==84.0.0" \
        "wheel==0.48.0" \
    && python -m build --wheel --no-isolation --outdir /wheels


FROM ${PYTHON_IMAGE} AS runtime

ARG PIP_VERSION=26.2.1
ARG CHERAGH_EXTRAS=fastapi,config

ENV CHERAGH_ENABLE_INDEXING=false \
    CHERAGH_INDEX=/data/.cheragh_index \
    CHERAGH_INDEX_ROOT=/data \
    CHERAGH_REQUIRE_AUTH=true \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY docker/constraints.txt /app/docker/constraints.txt
COPY --from=builder /wheels /tmp/wheels

RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install "pip==${PIP_VERSION}" \
    && wheel="$(find /tmp/wheels -maxdepth 1 -type f -name 'cheragh-*.whl' -print -quit)" \
    && test -n "${wheel}" \
    && /opt/venv/bin/python -m pip install \
        --constraint /app/docker/constraints.txt \
        "${wheel}[${CHERAGH_EXTRAS}]" \
    && /opt/venv/bin/python -m pip check \
    && rm -rf /tmp/wheels \
    && addgroup --system --gid 10001 cheragh \
    && adduser --system --uid 10001 --ingroup cheragh --no-create-home cheragh \
    && install -d -o cheragh -g cheragh -m 0750 /data

USER 10001:10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).read()"

STOPSIGNAL SIGTERM

CMD ["cheragh", "serve", "--index", "/data/.cheragh_index", "--host", "0.0.0.0", "--port", "8000", "--index-root", "/data"]
