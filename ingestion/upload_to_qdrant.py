"""
upload_to_qdrant.py — Upload embedded chunks to Qdrant Cloud.

Reads embedded_chunks.json, creates the Qdrant collection if it doesn't exist,
and upserts all vectors with their payloads (text + metadata for citations).

Also serializes a BM25 index to data/processed/bm25_index.pkl for sparse retrieval.

Environment variables required:
  QDRANT_URL            — e.g. https://xyz.us-east4-0.gcp.cloud.qdrant.io
  QDRANT_API_KEY        — from cloud.qdrant.io
  QDRANT_COLLECTION     — default: nps_docs
"""

import json
import os
import pickle
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from rank_bm25 import BM25Okapi
from tqdm import tqdm

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "nps_docs")

VECTOR_DIM = 1024       # bge-large-en-v1.5 output dimension
BATCH_SIZE = 100        # upsert batch size — Qdrant Cloud free tier is fine with this

DISTANCE_METRIC = qdrant_models.Distance.COSINE


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the collection if it doesn't already exist. Pass recreate=True to wipe and rebuild."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        if recreate:
            print(f"Deleting existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation")
            return

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_DIM,
            distance=DISTANCE_METRIC,
            on_disk=True,          # save memory on free tier
        ),
        optimizers_config=qdrant_models.OptimizersConfigDiff(
            indexing_threshold=10_000,   # build HNSW index after 10k vectors
        ),
    )
    print("Collection created.")


def upsert_chunks(client: QdrantClient, embedded_chunks: list[dict]) -> None:
    """Upsert all chunks in batches."""
    total = len(embedded_chunks)
    print(f"Uploading {total} vectors to Qdrant in batches of {BATCH_SIZE}...")

    for start in tqdm(range(0, total, BATCH_SIZE)):
        batch = embedded_chunks[start : start + BATCH_SIZE]
        points = []
        for chunk in batch:
            points.append(
                qdrant_models.PointStruct(
                    id=str(uuid4()),
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        "node_id": chunk["node_id"],
                        "title": chunk["title"],
                        "source_url": chunk["source_url"],
                        "doc_type": chunk["metadata"].get("doc_type", ""),
                        "parent_node_id": chunk.get("parent_node_id"),
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    print(f"Upserted {total} vectors successfully.")


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def build_and_save_bm25(embedded_chunks: list[dict]) -> None:
    """
    Build a BM25Okapi index over the chunk texts and save to disk.
    The retriever loads this file at startup.
    """
    print("Building BM25 index...")
    tokenized = [chunk["text"].lower().split() for chunk in embedded_chunks]
    bm25 = BM25Okapi(tokenized)

    # Also save the original texts + metadata so the retriever can map scores → chunks
    index_data = {
        "bm25": bm25,
        "chunks": [
            {
                "text": c["text"],
                "title": c["title"],
                "source_url": c["source_url"],
                "doc_type": c["metadata"].get("doc_type", ""),
                "node_id": c["node_id"],
            }
            for c in embedded_chunks
        ],
    }

    output_path = PROCESSED_DATA_DIR / "bm25_index.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(index_data, f)

    print(f"BM25 index saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_embedded_chunks(path: Path | None = None) -> list[dict]:
    if path is None:
        path = PROCESSED_DATA_DIR / "embedded_chunks.json"
    print(f"Loading embedded chunks from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_upload(embedded_chunks: list[dict] | None = None, recreate: bool = False) -> None:
    if embedded_chunks is None:
        embedded_chunks = load_embedded_chunks()

    client = get_client()
    ensure_collection(client, recreate=recreate)
    upsert_chunks(client, embedded_chunks)
    build_and_save_bm25(embedded_chunks)

    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection info: {info.vectors_count} vectors indexed")


if __name__ == "__main__":
    run_upload()
