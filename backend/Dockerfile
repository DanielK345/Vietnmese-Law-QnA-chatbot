# Build context is the backend/ directory.
# docker-compose handles this automatically via `context: .`
# Manual build from backend/:
#   docker build -t vn-law-backend .

FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Backend source + deps ──────────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -U -r requirements.txt

COPY . ./backend

# ── Runtime environment ────────────────────────────────────────────────────────
ENV PYTHONPATH="/app/backend/src"

WORKDIR /app/backend

# Default: production mode (entrypoint.sh decides api vs worker via CMD)
ENTRYPOINT ["bash", "entrypoint.sh"]
# Override CMD in docker-compose to select api vs worker mode
CMD []
