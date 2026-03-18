#!/usr/bin/env bash
# Safety flags (pipefail only if shell supports it)
set -eu
set -o pipefail 2>/dev/null || true

# Always run from backend root
cd "$(dirname "$0")"

# Ensure src and local FlagEmbedding are on PYTHONPATH
export PYTHONPATH="$PWD/src:$PWD/../retrieval/FlagEmbedding:${PYTHONPATH:-}"

DEBUG="${DEBUG:-false}"
CELERY_ENABLED="${CELERY_ENABLED:-true}"
PORT="${PORT:-8002}"

if [ "$DEBUG" = "true" ]; then
    echo "[entrypoint] DEBUG mode — uvicorn with --reload, no Celery worker"
    uvicorn app:app \
        --app-dir src \
        --host 0.0.0.0 \
        --port "$PORT" \
        --reload \
        --log-level debug
    exit 0
fi

# When running in Docker, docker-compose passes "worker" as the first argument
# to start a dedicated Celery worker container instead of the API.
if [ "${1:-}" = "worker" ]; then
    echo "[entrypoint] Celery worker mode"
    exec celery -A tasks.celery_app worker --loglevel=info
fi

if [ "$CELERY_ENABLED" = "true" ]; then
    echo "[entrypoint] Production mode — uvicorn + Celery worker (single container)"
    python src/app.py &
    celery -A src.tasks.celery_app worker --loglevel=info
else
    echo "[entrypoint] Uvicorn-only mode (CELERY_ENABLED=false) — no Celery worker"
    python src/app.py
fi

