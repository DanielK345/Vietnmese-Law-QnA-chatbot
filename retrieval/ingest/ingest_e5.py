"""
Ingest corpus.csv into Qdrant Cloud — Multilingual-E5-large collection (dense only).

Usage:
    CUDA_VISIBLE_DEVICES=0 python ingest_e5.py --csv ../data/corpus.csv --batch-size 64
"""
import argparse
import json
import os
import re
import time
import uuid

import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "backend", ".env"))

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTIONS = [c.strip() for c in os.getenv("COLLECTIONS", "vn_law_bge_m3,vn_law_e5").split(",")]
E5_COLLECTION = COLLECTIONS[1]

DENSE_DIM = 1024  # multilingual-e5-large output dimension


def _detect_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            print(f"[Device] GPU detected: {name} ({vram:.1f} GB VRAM) — using CUDA")
            return "cuda"
        print("[Device] No CUDA GPU detected — falling back to CPU (ingestion will be slow)")
        return "cpu"
    except ImportError:
        print("[Device] torch not installed — using CPU")
        return "cpu"


DEVICE = _detect_device()

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checkpoints")


def _checkpoint_path(collection_name: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{collection_name}.ckpt.json")


def save_checkpoint(collection_name: str, chunks_done: int, total_chunks: int) -> None:
    """Persist ingestion progress so runs can be resumed after interruption."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    with open(_checkpoint_path(collection_name), "w") as f:
        json.dump({"chunks_done": chunks_done, "total_chunks": total_chunks}, f)


def load_checkpoint(collection_name: str) -> int:
    """Return the chunk index to start from (0 if no checkpoint exists)."""
    path = _checkpoint_path(collection_name)
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        data = json.load(f)
    resume_from = data["chunks_done"]
    total = data["total_chunks"]
    if total > 0 and resume_from >= total:
        print(f"[Checkpoint] Already fully ingested ({total:,} chunks) — skipping.")
    else:
        print(f"[Checkpoint] Resuming from chunk {resume_from:,} / {total:,}")
    return resume_from


def split_text_keeping_sentences(text: str, max_word_count: int = 400) -> list[str]:
    """Split text into chunks of at most max_word_count words, preserving sentence boundaries."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk, current_wc = [], "", 0
    for sentence in sentences:
        wc = len(sentence.split())
        if current_wc + wc > max_word_count and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk, current_wc = sentence, wc
        else:
            current_chunk += (" " + sentence.strip()) if current_chunk else sentence.strip()
            current_wc += wc
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def create_collection_if_needed(client: QdrantClient, collection_name: str):
    """Create the E5 dense-only collection if it doesn't already exist."""
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — skipping creation.")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
    )
    print(f"Created collection '{collection_name}'.")


def _print_progress(desc: str, n: int, total: int, t_start: float) -> None:
    elapsed = time.time() - t_start
    pct = 100.0 * n / total if total else 0.0
    rate = n / elapsed if elapsed > 0 else 0.0
    eta = (total - n) / rate if rate > 0 else float("inf")
    eta_str = f"~{int(eta)}s" if eta != float("inf") else "?"
    print(f"  [{desc}] {n:,}/{total:,} ({pct:.1f}%) | {elapsed:.1f}s elapsed | ETA {eta_str}", flush=True)


def ingest(csv_path: str, batch_size: int = 64):
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    create_collection_if_needed(client, E5_COLLECTION)

    model = SentenceTransformer("intfloat/multilingual-e5-large", device=DEVICE)
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Chunk all rows
    all_chunks = []  # list of (text, infor_id, chunk_id)
    chunk_id = 0
    n_rows = len(df)
    _print_every = max(1, n_rows // 20)
    t_chunk = time.time()
    for n_row, row in enumerate(df.itertuples(index=False), 1):
        text, cid = row.text, row.cid
        if not isinstance(text, str) or not text.strip():
            continue
        for chunk_text in split_text_keeping_sentences(text, max_word_count=400):
            all_chunks.append((chunk_text, int(cid), chunk_id))
            chunk_id += 1
        if n_row % _print_every == 0 or n_row == n_rows:
            _print_progress("Chunking", n_row, n_rows, t_chunk)

    print(f"Total chunks to ingest: {len(all_chunks)}")

    # Batch encode and upsert (checkpoint-aware)
    start_from = load_checkpoint(E5_COLLECTION)
    if start_from >= len(all_chunks):
        print(f"All {len(all_chunks):,} chunks already ingested — nothing to do.")
        return

    _batch_indices = range(start_from, len(all_chunks), batch_size)
    _n_batches = len(_batch_indices)
    t_encode = time.time()
    for _batch_num, i in enumerate(_batch_indices, 1):
        batch = all_chunks[i : i + batch_size]
        # E5 uses "passage: " prefix for documents
        texts = ["passage: " + c[0] for c in batch]

        dense_vecs = model.encode(texts, normalize_embeddings=True)

        points = []
        for j, (text, infor_id, c_id) in enumerate(batch):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"dense": dense_vecs[j].tolist()},
                    payload={"text": text, "infor_id": infor_id, "chunk_id": c_id},
                )
            )

        client.upsert(collection_name=E5_COLLECTION, points=points)
        save_checkpoint(E5_COLLECTION, i + len(batch), len(all_chunks))
        _print_progress("Encoding & upserting", _batch_num, _n_batches, t_encode)

    # Mark checkpoint as complete so re-runs skip this collection
    save_checkpoint(E5_COLLECTION, len(all_chunks), len(all_chunks))
    print(f"Done — {len(all_chunks)} points upserted into '{E5_COLLECTION}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest corpus into E5 Qdrant collection")
    parser.add_argument("--csv", default="../data/corpus.csv", help="Path to corpus.csv")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding + upsert batch size")
    args = parser.parse_args()
    ingest(args.csv, args.batch_size)
