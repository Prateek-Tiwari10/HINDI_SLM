# Demo server image. Build from the repo root:
#   docker build -f serve/Dockerfile -t hinglish-translit .
#   docker run --rm -p 8000:8000 --cpus 0.5 --memory 1g hinglish-translit
# Weights are baked in at build time from the pinned revision in serve/MODEL_ID (or --build-arg MODEL_ID=...),
# so a cold start never downloads anything.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TORCH_THREADS=1

WORKDIR /app

# 1. dependencies (CPU-only torch wheel: no CUDA libraries in the image)
COPY serve/requirements.txt serve/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install -r serve/requirements.txt

# 2. weights from the pinned HF revision
ARG MODEL_ID=""
COPY serve/__init__.py serve/model_ref.py serve/MODEL_ID serve/
RUN MODEL_ID="${MODEL_ID}" python -m serve.model_ref
ENV MODEL_ID=${MODEL_ID}

# 3. code (only what inference imports; see .dockerignore)
COPY data/__init__.py data/normalize.py data/
COPY tokenizer/__init__.py tokenizer/hinglish.py tokenizer/
COPY model/ model/
COPY finetune/__init__.py finetune/data.py finetune/translit.py finetune/
COPY serve/ serve/

ARG GIT_SHA=dev
ENV GIT_SHA=${GIT_SHA}
LABEL org.opencontainers.image.source="https://github.com/Prateek-Tiwari10/HINDI_SLM" \
      org.opencontainers.image.description="Hinglish to Devanagari demo (61M model, CPU)"

RUN useradd --create-home --uid 10001 app && chown -R app /opt/hf
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status != 200)"
# uvicorn's own text logs are cut to warnings; the app writes one JSON line per request (with latency_ms)
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--log-level", "warning"]
