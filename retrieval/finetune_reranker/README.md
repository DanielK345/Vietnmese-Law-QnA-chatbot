# Reranker Fine-tuning

Fine-tunes **[BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)** on the Vietnamese legal corpus using hard-negative mining from both Qdrant collections (BGE-M3 hybrid + E5 dense).

---

## Pipeline Overview

```
train.csv
    │
    ├─ [1] hard_negative_bge_round1.py  ──► data_round1.json
    │       (BGE-M3 hybrid search, RRF)
    │
    ├─ [2] hard_negative_e5.py          ──► save_pairs_e5_top25.pkl
    │       (E5 dense search)
    │
    └─ [3] create_data_rerank.py        ──► data_reranking.json
            (merge BGE + E5 negatives,
             deduplicate by chunk_id)
                │
                └─ [4] finetune.sh      ──► fine-tuned reranker checkpoint
```

---

## Prerequisites

### 1. Environment

Run `setup_env.sh` from inside this folder to create a dedicated venv with FlagEmbedding (finetune extras) and PyTorch (CUDA 12.1):

```bash
cd retrieval/finetune_reranker
bash setup_env.sh
source .venv/bin/activate
```

> **Windows:** Run these steps manually — create the venv, install FlagEmbedding from `../FlagEmbedding` with `pip install -e .[finetune]`, then install torch from `https://download.pytorch.org/whl/cu121`.

### 2. Input Data

`train.csv` must contain the following columns:

| Column | Type | Description |
|---|---|---|
| `question` | string | Query / legal question |
| `cid` | string | Space-separated list of ground-truth document IDs, e.g. `[42 87 103]` |
| `context` | string | Python-literal list of ground-truth passage strings |

### 3. Qdrant Collections

Both Qdrant Cloud collections (`vn_law_bge_m3` and `vn_law_e5`) must already be populated. Run the [ingestion scripts](../README.md) first if they are empty.

The scripts currently hard-code `host="http://localhost:6333"` and legacy collection names. **Update the `__main__` blocks** in each script to match your Qdrant Cloud credentials from `backend/.env`:

```python
# Replace in hard_negative_bge_round1.py __main__
qdrant_search = QdrantSearch_bge(
    host="https://<your-cluster>.aws.cloud.qdrant.io",   # QDRANT_URL from .env
    collection_name="vn_law_bge_m3",
    model_name="BAAI/bge-m3",
    use_fp16=True
)

# Replace in hard_negative_e5.py __main__
qdrant_search = QdrantSearch_e5(
    host="https://<your-cluster>.aws.cloud.qdrant.io",   # QDRANT_URL from .env
    collection_name="vn_law_e5",
    model_name="intfloat/multilingual-e5-large",
)

# Replace in create_data_rerank.py __main__
qdrant_search_bge = QdrantSearch_bge(host=..., collection_name="vn_law_bge_m3", ...)
qdrant_search_e5  = QdrantSearch_e5(host=...,  collection_name="vn_law_e5", ...)
```

> `QdrantClient` accepts an `api_key=` keyword argument for authenticated Qdrant Cloud clusters.

---

## Step-by-Step

### Step 1 — Hard-negative mining with BGE-M3

```bash
python hard_negative_bge_round1.py
```

**What it does:**
- Loads each row from `train.csv`.
- Chunks positive ground-truth passages (≤ 400 words each, sentence-boundary aware).
- Retrieves the top-25 candidates from `vn_law_bge_m3` via BGE-M3 hybrid search (dense + sparse → RRF fusion).
- Any retrieved result whose `infor_id` is **not** in the positive IDs is treated as a hard negative.

**Output:** `data_round1.json` — one JSON object per line in FlagEmbedding training format:
```json
{"query": "...", "pos": ["passage A", "passage B"], "neg": ["hard neg 1", ...]}
```

**Key configuration** (in `__main__`):

| Variable | Default | Description |
|---|---|---|
| `csv_path` | `train.csv` | Input training CSV |
| `output_path` | `...` | Directory for `data_round1.json` |
| `limit` | 25 | Top-k candidates retrieved per query |

---

### Step 2 — Hard-negative mining with E5

```bash
python hard_negative_e5.py
```

**What it does:**
- Same positive/negative split logic as Step 1 but uses E5 dense search (`query: ` prefix).
- Saves individual positive and negative pairs with a `relevant` label (1 / 0).

**Output:** `save_pairs_e5_top25.pkl` — Python pickle list of dicts:
```python
{"question": "query: ...", "document": "passage: ...", "relevant": 1}  # positive
{"question": "query: ...", "document": "passage: ...", "relevant": 0}  # negative
```

**Key configuration** (in `__main__`):

| Variable | Default | Description |
|---|---|---|
| `csv_path` | `train.csv` | Input training CSV |
| `output_path` | `...` | Directory for the pickle file |
| `limit` | 25 | Top-k candidates retrieved per query |

---

### Step 3 — Combine negatives for reranker training data

```bash
python create_data_rerank.py
```

**What it does:**
- Queries **both** `vn_law_bge_m3` (top-25) and `vn_law_e5` (top-25) for each question.
- Merges the negative pools, deduplicating by `chunk_id` so no passage appears twice.
- Produces a single combined training file for the reranker.

**Output:** `data_reranking.json` — same line-delimited JSON format as Step 1:
```json
{"query": "...", "pos": ["..."], "neg": ["hard neg from BGE", "hard neg from E5", ...]}
```

**Key configuration** (in `__main__`):

| Variable | Default | Description |
|---|---|---|
| `csv_path` | `train_data.csv` | Input training CSV |
| `output_path` | `/format_data/rerank` | Output directory |

---

### Step 4 — Fine-tune the reranker

```bash
bash finetune.sh
```

**What it does:**  
Runs `torchrun` with a single GPU to fine-tune `BAAI/bge-reranker-v2-m3` using FlagEmbedding's `encoder_only.base` trainer.

**Key hyperparameters:**

| Argument | Value | Description |
|---|---|---|
| `--model_name_or_path` | `BAAI/bge-reranker-v2-m3` | Base reranker model |
| `--train_data` | `data_reranking.json` | Training file from Step 3 |
| `--train_group_size` | 8 | Positives + negatives per query in a group |
| `--query_max_len` | 128 | Max tokens for the query |
| `--passage_max_len` | 400 | Max tokens per passage (matches chunking limit) |
| `--learning_rate` | 1e-5 | AdamW learning rate |
| `--num_train_epochs` | 6 | Training epochs |
| `--per_device_train_batch_size` | 2 | Per-GPU batch size |
| `--gradient_accumulation_steps` | 2 | Effective batch = 2 × 2 = 4 |
| `--warmup_ratio` | 0.1 | Linear warmup over 10% of steps |
| `--fp16` | ✓ | Mixed-precision training |
| `--gradient_checkpointing` | ✓ | Reduces VRAM at the cost of speed |
| `--deepspeed` | `ds_stage0.json` | DeepSpeed ZeRO Stage 0 |
| `--output_dir` | `output_folder` | Where checkpoints are saved |

**Before running**, update the paths in `finetune.sh`:
```bash
--train_data /path/to/data_reranking.json \
--output_dir /path/to/reranker_checkpoint \
```

**GPU requirement:** At least one CUDA GPU with ~16 GB VRAM recommended (`bge-reranker-v2-m3` is ~560M parameters). Gradient checkpointing and fp16 reduce peak VRAM significantly.

---

## File Reference

| File | Role |
|---|---|
| `setup_env.sh` | Creates venv, installs FlagEmbedding `[finetune]` + PyTorch CUDA 12.1 |
| `hard_negative_bge_round1.py` | Mines hard negatives using BGE-M3 hybrid search → `data_round1.json` |
| `hard_negative_e5.py` | Mines hard negatives using E5 dense search → `save_pairs_e5_top25.pkl` |
| `create_data_rerank.py` | Combines BGE + E5 negatives → `data_reranking.json` |
| `finetune.sh` | Fine-tunes `bge-reranker-v2-m3` on `data_reranking.json` |

---

## Notes

- **Why two models for hard-negative mining?** BGE-M3 (hybrid) and E5 (dense) have different retrieval characteristics. Combining their top-k candidates gives a more diverse and challenging set of negatives, which produces a stronger reranker.
- **`train_group_size 8`** means each training sample consists of 1 query, positives from `pos`, and enough negatives drawn from `neg` to fill a group of 8. Having ≥ 7 negatives per query in `data_reranking.json` is recommended.
- **Checkpoints** are saved every 29,000 steps (`--save_steps 29000`), keeping the last 8 (`--save_total_limit 8`).
- The `ds_stage0.json` DeepSpeed config from the parent FlagEmbedding repo is required — copy it from `retrieval/FlagEmbedding/examples/finetune/embedder/ds_stage0.json` into this folder before running.
