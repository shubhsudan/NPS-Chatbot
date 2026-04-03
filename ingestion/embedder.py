"""
embedder.py — Embed leaf chunks using bge-large-en-v1.5.

Reads leaf_chunks.json, generates a 1024-dim embedding per chunk,
and writes embedded_chunks.json where each record adds an "embedding" field.

bge-large-en-v1.5 notes:
  - Prefix queries with "Represent this sentence for searching relevant passages: "
    at inference time (query-side only, not document-side)
  - Documents are embedded as-is
  - 1024 dimensions, strong on formal/regulatory English text
"""

import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32  # adjust down if you hit OOM on CPU


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def load_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return model


def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> list[dict]:
    """
    Add an "embedding" field (list[float]) to each chunk dict.
    Returns the enriched list.
    """
    texts = [chunk["text"] for chunk in chunks]

    print(f"Embedding {len(texts)} chunks in batches of {batch_size}...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity works on unit vectors
        convert_to_numpy=True,
    )

    embedded = []
    for chunk, vec in zip(chunks, embeddings):
        record = dict(chunk)
        record["embedding"] = vec.tolist()
        embedded.append(record)

    return embedded


def save_embeddings(embedded_chunks: list[dict]) -> Path:
    output_path = PROCESSED_DATA_DIR / "embedded_chunks.json"
    print(f"Saving {len(embedded_chunks)} embedded chunks → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False)
    return output_path


def load_leaf_chunks(path: Path | None = None) -> list[dict]:
    if path is None:
        path = PROCESSED_DATA_DIR / "leaf_chunks.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_embedder(chunks: list[dict] | None = None) -> list[dict]:
    """
    Run embedder. If chunks is None, loads from data/processed/leaf_chunks.json.
    Returns embedded_chunks list.
    """
    if chunks is None:
        print("Loading leaf chunks from disk...")
        chunks = load_leaf_chunks()

    model = load_model()
    embedded = embed_chunks(chunks, model)
    save_embeddings(embedded)

    print(f"\nDone. Embedding dimension: {len(embedded[0]['embedding'])}")
    return embedded


if __name__ == "__main__":
    run_embedder()
