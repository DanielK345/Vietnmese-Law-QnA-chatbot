#!/bin/bash
# Ingest corpus data into both Qdrant Cloud collections.
# Run from the repository root:  sh retrieval/ingest.sh

set -e

CSV_PATH="${CSV_PATH:-data/corpus.csv}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Auto-detect GPU via nvidia-smi; Python scripts fall back to CPU automatically
GPU_INFO=""
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
fi

if [ -n "$GPU_INFO" ]; then
    echo "[Device] GPU detected: $GPU_INFO"
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE:-0}"
else
    echo "[Device] No GPU detected — running on CPU (ingestion will be slow)"
fi

# Resolve python: prefer python3, fall back to python
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null || echo "")
if [ -z "$PYTHON" ]; then
    echo "ERROR: No python interpreter found. Activate your venv or install Python."
    exit 1
fi

echo "=== Ingestion config ==="
echo "Python:     $PYTHON"
echo "CSV:        $CSV_PATH"
echo "Batch size: $BATCH_SIZE"
echo "========================"

cd "$(dirname "$0")"

echo ""
echo ">>> [1/2] Ingesting into BGE-m3 collection..."
"$PYTHON" ingest_bge.py --csv "../$CSV_PATH" --batch-size "$BATCH_SIZE"

echo ""
echo ">>> [2/2] Ingesting into E5 collection..."
"$PYTHON" ingest_e5.py --csv "../$CSV_PATH" --batch-size "$BATCH_SIZE"

echo ""
echo "=== All ingestion complete ==="
