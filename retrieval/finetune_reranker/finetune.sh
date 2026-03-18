#!/bin/bash
# Unified reranker fine-tuning pipeline
# Replaces setup_env.sh + the original finetune.sh
#
# Steps:
#   0. One-time venv setup (skipped when .venv already exists)
#   1. Hard-negative mining  — BGE-M3 hybrid search  → data_round1.json
#   2. Hard-negative mining  — E5 dense search        → save_pairs_e5_top25.pkl
#   3. Combine negatives     → data_reranking.json
#   4. Fine-tune bge-reranker-v2-m3
#
# Usage:
#   bash finetune.sh
#
# Override defaults via env vars:
#   TRAIN_CSV=my_data.csv OUTPUT_PATH=./data EPOCHS=3 bash finetune.sh

set -e

# ── Configuration ──────────────────────────────────────────────────────────
TRAIN_CSV="${TRAIN_CSV:-train.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-./output_data}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-./reranker_checkpoint}"
CACHE_DIR="${CACHE_DIR:-./data_for_finetune}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
FLAG_EMBEDDING_DIR="${FLAG_EMBEDDING_DIR:-../FlagEmbedding}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-ds_stage0.json}"

# Export so Python sub-scripts can read them
export TRAIN_CSV OUTPUT_PATH

# ── Step 0: One-time environment setup ─────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "=== [0/4] Setting up Python virtual environment ==="
    python -m venv .venv
    source .venv/bin/activate

    echo "Installing FlagEmbedding with finetune extras..."
    pip install -e "$FLAG_EMBEDDING_DIR[finetune]"

    echo "Installing PyTorch with CUDA 12.1 support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    echo "Environment setup complete."
else
    echo "=== [0/4] Activating existing virtual environment ==="
    source .venv/bin/activate
fi

# ── Ensure DeepSpeed config is available ───────────────────────────────────
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
    echo "WARNING: $DEEPSPEED_CONFIG not found — copying from FlagEmbedding examples"
    cp "$FLAG_EMBEDDING_DIR/examples/finetune/embedder/ds_stage0.json" "$DEEPSPEED_CONFIG" 2>/dev/null || \
        { echo "ERROR: ds_stage0.json not found. Set DEEPSPEED_CONFIG env var."; exit 1; }
fi

mkdir -p "$OUTPUT_PATH"

# ── Step 1: Hard-negative mining — BGE-M3 ──────────────────────────────────
echo ""
echo "=== [1/4] Hard-negative mining with BGE-M3 ==="
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python hard_negative_bge_round1.py

# ── Step 2: Hard-negative mining — E5 ──────────────────────────────────────
echo ""
echo "=== [2/4] Hard-negative mining with E5 ==="
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python hard_negative_e5.py

# ── Step 3: Create combined reranker training data ──────────────────────────
echo ""
echo "=== [3/4] Creating combined reranker training data ==="
python create_data_rerank.py

# ── Step 4: Fine-tune bge-reranker-v2-m3 ───────────────────────────────────
echo ""
echo "=== [4/4] Fine-tuning bge-reranker-v2-m3 ==="
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=$CUDA_DEVICE \
torchrun --nproc_per_node 1 --master_port 29502 \
    -m FlagEmbedding.finetune.reranker.encoder_only.base \
    --model_name_or_path BAAI/bge-reranker-v2-m3 \
    --train_data "$OUTPUT_PATH/data_reranking.json" \
    --cache_path "$CACHE_DIR" \
    --train_group_size 8 \
    --query_max_len 128 \
    --passage_max_len 400 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --output_dir "$TRAIN_OUTPUT_DIR" \
    --bge_m3_v2_with_random \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --logging_steps 29000 \
    --save_steps 29000 \
    --save_total_limit 8

echo ""
echo "=== Pipeline complete ==="
echo "Fine-tuned checkpoint → $TRAIN_OUTPUT_DIR"