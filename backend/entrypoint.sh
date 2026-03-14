#!/usr/bin/env bash
# Safety flags (pipefail only if shell supports it)
set -eu
set -o pipefail 2>/dev/null || true

# Always run from backend root
cd "$(dirname "$0")"

# Ensure src and local FlagEmbedding are on PYTHONPATH
export PYTHONPATH="$PWD/src:$PWD/../retrieval/FlagEmbedding:${PYTHONPATH:-}"

# Start FastAPI app in background
python src/app.py &

# Start Celery worker
celery -A src.tasks.celery_app worker --loglevel=debug

