# syntax=docker/dockerfile:1.7
# ---------- builder: compile wheels once, keep the runtime image slim ----------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    DB_PATH=/data/mqmas.db \
    PORT=8000 \
    MALLOC_ARENA_MAX=2
RUN useradd --create-home --uid 10001 quant \
 && mkdir -p /data && chown quant:quant /data
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=quant:quant mqmas ./mqmas
USER quant
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health', timeout=4)" || exit 1
# Single worker on purpose: the trading loop must exist exactly once per process.
CMD ["sh", "-c", "exec uvicorn mqmas.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
