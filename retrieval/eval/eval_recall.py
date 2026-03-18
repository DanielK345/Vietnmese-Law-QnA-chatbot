"""
Evaluate Recall@k for BGE-m3, E5, and Combined (BGE+E5) retrieval.

The script samples N rows from train.csv (which carries ground-truth cid labels),
runs each query against the live Qdrant Cloud collections, then reports Recall@k
for k in {3, 5, 10} (or any values you pass with --k).

Usage (from repo root, with venv active):
    python retrieval/eval/eval_recall.py
    python retrieval/eval/eval_recall.py --n-samples 200 --k 3 5 10 --seed 0

Requirements:
    - backend/.env must contain QDRANT_URL, QDRANT_API_KEY, COLLECTIONS
    - Both Qdrant collections must already be populated (run retrieval/ingest/)
    - GPU recommended (falls back to CPU automatically)
"""

import argparse
import ast
import os
import sys
import time
from collections import defaultdict

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — add backend/src so search_document imports resolve
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))          # repo root
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, "backend", ".env"))

from search_document.search_with_bge import QdrantSearch_bge  # noqa: E402
from search_document.search_with_e5 import QdrantSearch_e5    # noqa: E402


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_eval_samples(csv_path: str, n_samples: int, seed: int) -> list[dict]:
    """
    Return a list of dicts with keys:
        query         : str
        relevant_cids : set[int]

    Rows whose 'cid' column cannot be parsed are silently skipped.
    """
    df = pd.read_csv(csv_path)
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    samples = []
    for _, row in df.iterrows():
        try:
            relevant_cids = set(ast.literal_eval(row["cid"]))
        except Exception:
            continue
        samples.append({"query": str(row["question"]), "relevant_cids": relevant_cids})
    return samples


# ---------------------------------------------------------------------------
# Search helpers
# ---------------------------------------------------------------------------

def get_infor_ids(results) -> list[int]:
    """Extract ordered list of infor_id from a Qdrant QueryResponse."""
    return [int(p.payload["infor_id"]) for p in results.points]


def combined_ids(bge_ids: list[int], e5_ids: list[int]) -> list[int]:
    """
    Merge BGE and E5 result lists, preserving order and deduplicating.
    BGE results come first; E5 additions are appended in their original order.
    """
    seen: set[int] = set()
    merged: list[int] = []
    for cid in bge_ids + e5_ids:
        if cid not in seen:
            seen.add(cid)
            merged.append(cid)
    return merged


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> bool:
    """Return True if any of the top-k retrieved ids is in the relevant set."""
    return bool(set(retrieved[:k]) & relevant)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    samples: list[dict],
    bge_search: QdrantSearch_bge,
    e5_search: QdrantSearch_e5,
    k_values: list[int],
) -> tuple[dict, int]:
    """
    Run every query through BGE, E5, and Combined, accumulate Recall@k hits.

    Returns:
        scores : {model_key: {k: recall_pct}}
        total  : number of evaluated samples
    """
    max_k = max(k_values)
    hits: dict[str, dict[int, int]] = {
        "bge": defaultdict(int),
        "e5":  defaultdict(int),
        "combined": defaultdict(int),
    }
    n_total = len(samples)

    for i, sample in enumerate(samples, 1):
        query = sample["query"]
        relevant = sample["relevant_cids"]

        bge_res = get_infor_ids(bge_search.search(query, limit=max_k))
        e5_res  = get_infor_ids(e5_search.search(query, limit=max_k))
        comb    = combined_ids(bge_res, e5_res)

        for k in k_values:
            hits["bge"][k]      += recall_at_k(bge_res, relevant, k)
            hits["e5"][k]       += recall_at_k(e5_res,  relevant, k)
            hits["combined"][k] += recall_at_k(comb,    relevant, k)

        if i % 50 == 0 or i == n_total:
            print(f"  [{i:>{len(str(n_total))}}/{n_total}] queries evaluated", flush=True)

    scores = {
        model: {k: 100.0 * count / n_total for k, count in k_hits.items()}
        for model, k_hits in hits.items()
    }
    return scores, n_total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_MODEL_LABELS = [
    ("bge",      "BGE-m3"),
    ("e5",       "E5"),
    ("combined", "Combined (BGE+E5)"),
]


def print_table(scores: dict, k_values: list[int], n_total: int, elapsed: float) -> None:
    col_w = 12
    label_w = 22
    sep = "=" * (label_w + col_w * len(k_values))

    print()
    print(sep)
    print(f"Recall@k  —  n={n_total} samples  |  {elapsed:.1f}s")
    print(sep)
    header = f"{'Model':<{label_w}}" + "".join(f"{'K='+str(k):>{col_w}}" for k in k_values)
    print(header)
    print("-" * (label_w + col_w * len(k_values)))
    for key, label in _MODEL_LABELS:
        row = f"{label:<{label_w}}" + "".join(
            f"{scores[key][k]:>{col_w-1}.2f}%" for k in k_values
        )
        print(row)
    print(sep)

    # Improvement rows (combined vs single models)
    print(f"\n{'Delta: Combined vs BGE':<{label_w}}" + "".join(
        f"{scores['combined'][k]-scores['bge'][k]:>{col_w-1}.2f}%" for k in k_values
    ))
    print(f"{'Delta: Combined vs E5':<{label_w}}" + "".join(
        f"{scores['combined'][k]-scores['e5'][k]:>{col_w-1}.2f}%" for k in k_values
    ))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Recall@k: BGE-m3 vs E5 vs Combined (BGE+E5)"
    )
    parser.add_argument(
        "--csv",
        default="data/train.csv",
        help="CSV with ground-truth labels (needs 'question' and 'cid' columns). "
             "Default: data/train.csv",
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000,
        help="Number of rows to sample for evaluation (default: 1000)",
    )
    parser.add_argument(
        "--k", type=int, nargs="+", default=[3, 5, 10],
        help="K values for Recall@k (default: 3 5 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(ROOT, args.csv)

    # ---- configuration summary ----
    collections_env = os.getenv("COLLECTIONS", "vn_law_bge_m3,vn_law_e5")
    collections = [c.strip() for c in collections_env.split(",")]
    bge_collection = collections[0]
    e5_collection  = collections[1]

    print("=" * 60)
    print("Retrieval Recall@k Evaluation")
    print("=" * 60)
    print(f"  CSV          : {csv_path}")
    print(f"  Samples      : {args.n_samples}  |  Seed: {args.seed}")
    print(f"  K values     : {k_values}")
    print(f"  BGE collection: {bge_collection}")
    print(f"  E5  collection: {e5_collection}")
    print("=" * 60)

    # ---- load data ----
    print("\nLoading eval samples...")
    samples = load_eval_samples(csv_path, args.n_samples, args.seed)
    print(f"  {len(samples)} valid samples loaded\n")

    # ---- init models ----
    print("Initialising BGE-m3 search...")
    bge_search = QdrantSearch_bge(
        collection_name=bge_collection,
        model_name="BAAI/bge-m3",
        use_fp16=True,
    )

    print("Initialising E5 search...")
    e5_search = QdrantSearch_e5(
        collection_name=e5_collection,
        model_name="intfloat/multilingual-e5-large",
        use_fp16=True,
    )

    # ---- evaluate ----
    print(f"\nRunning evaluation ({len(samples)} queries)...\n")
    t0 = time.time()
    scores, n_total = evaluate(samples, bge_search, e5_search, k_values)
    elapsed = time.time() - t0

    # ---- report ----
    print_table(scores, k_values, n_total, elapsed)


if __name__ == "__main__":
    main()
