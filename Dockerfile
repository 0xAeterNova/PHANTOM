# syntax=docker/dockerfile:1

FROM python:3.14-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip wheel --wheel-dir /wheels ".[api]"


FROM python:3.14-slim-bookworm AS app-base

LABEL org.opencontainers.image.title="Project PHANTOM" \
      org.opencontainers.image.description="Local consent-first browser research prototype"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PHANTOM_LOG_LEVEL=INFO \
    HOME=/var/lib/phantom \
    PHANTOM_CACHE_DIR=/var/lib/phantom \
    HF_HOME=/var/lib/phantom/huggingface \
    XDG_CACHE_HOME=/var/lib/phantom/cache \
    DEEPFACE_HOME=/var/lib/phantom \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1

RUN groupadd --gid 10001 phantom \
    && useradd --uid 10001 --gid phantom --home-dir /var/lib/phantom --no-create-home --shell /usr/sbin/nologin phantom \
    && install -d -o phantom -g phantom /var/lib/phantom

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN python -m pip install --no-index --find-links=/wheels "project-phantom[api]" \
    && rm -rf /wheels

COPY --chown=phantom:phantom app ./app
COPY --chown=phantom:phantom configs ./configs
COPY --chown=phantom:phantom prompts ./prompts
COPY --chown=phantom:phantom scripts/docker_entrypoint.py scripts/container_smoke.py scripts/check_realtime_models.py scripts/setup_piper_tts.py ./scripts/
COPY LICENSE /app/LICENSE

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"]

# Only the container binds all interfaces. Compose publishes it on host loopback.
# Keep one worker: sessions and models live in this process's memory.
ENTRYPOINT ["python", "scripts/docker_entrypoint.py"]
CMD ["python", "-m", "uvicorn", "app.realtime:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers"]


FROM app-base AS realtime

# CPU-only PyTorch avoids the multi-gigabyte CUDA dependency set from PyPI.
ARG TORCH_VERSION=2.14.1+cpu
COPY requirements-docker.txt ./
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates espeak-ng libgl1 libglib2.0-0 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==${TORCH_VERSION}" \
    && python -m pip install -r requirements-docker.txt \
    && python -m pip check

COPY docs/third_party_tts.md /app/docs/third_party_tts.md

ENV PHANTOM_IMAGE_MODE=real \
    PHANTOM_CONFIG=/app/configs/realtime_cpu.yaml \
    CUDA_VISIBLE_DEVICES="" \
    TF_USE_LEGACY_KERAS=1 \
    USE_TF=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

USER 10001:10001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3).read()"]


# Keep demo last so plain `docker build .` never pulls in the ML stack.
FROM app-base AS demo

ENV PHANTOM_IMAGE_MODE=demo \
    PHANTOM_CONFIG=/app/configs/realtime_demo.yaml

USER 10001:10001
