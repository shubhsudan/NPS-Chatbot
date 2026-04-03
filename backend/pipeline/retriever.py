"""
retriever.py — Hybrid retrieval: dense (Qdrant) + sparse (BM25), merged via RRF.

Public API:
    retriever = HybridRetriever()
    results   = retriever.retrieve("Can I withdraw NPS corpus before 60?", top_k=5)

Each result is a RetrievedChunk with text, title, source_url, and rrf_score.
No LLM is involved here — pure retrieval only.
"""

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "nps_docs")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")

# Path to BM25 index built by upload_to_qdrant.py
BM25_INDEX_PATH = (
    Path(__file__).parent.parent.parent / "data" / "processed" / "bm25_index.pkl"
)

# BGE requires this prefix on the query side (not on documents)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# RRF constant — k=60 is the standard default from the original paper
RRF_K = 60

DEFAULT_TOP_K = 5
DENSE_FETCH = 20   # fetch more candidates before merging so RRF has room to rerank
SPARSE_FETCH = 20


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    text: str
    title: str
    source_url: str
    doc_type: str
    node_id: str
    rrf_score: float
    dense_rank: Optional[int]   # 1-based rank in dense list, None if not retrieved
    sparse_rank: Optional[int]  # 1-based rank in sparse list, None if not retrieved


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Loads the embedding model, Qdrant client, and BM25 index once at startup,
    then serves retrieve() calls cheaply.
    """

    def __init__(self) -> None:
        self._embed_model = self._load_embed_model()
        self._qdrant = self._load_qdrant()
        self._bm25_data = self._load_bm25()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_embed_model(self) -> SentenceTransformer:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever] Loading embedding model ({EMBED_MODEL_NAME}) on {device}...")
        return SentenceTransformer(EMBED_MODEL_NAME, device=device)

    def _load_qdrant(self) -> QdrantClient:
        print(f"[Retriever] Connecting to Qdrant at {QDRANT_URL}...")
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

    def _load_bm25(self) -> dict:
        """Load the pickled BM25 index + chunk list from disk."""
        print(f"[Retriever] Loading BM25 index from {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Dense retrieval
    # ------------------------------------------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int,
        precomputed_embedding: list[float] | None = None,
    ) -> list[dict]:
        """Embed query (or use a precomputed vector) and search Qdrant."""
        if precomputed_embedding is not None:
            query_vec = precomputed_embedding
        else:
            query_vec = self._embed_model.encode(
                BGE_QUERY_PREFIX + query,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).tolist()

        hits = self._qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": h.payload["text"],
                "title": h.payload.get("title", ""),
                "source_url": h.payload.get("source_url", ""),
                "doc_type": h.payload.get("doc_type", ""),
                "node_id": h.payload.get("node_id", ""),
                "score": h.score,
            }
            for h in hits
        ]

    # ------------------------------------------------------------------
    # Sparse retrieval
    # ------------------------------------------------------------------

    def _sparse_search(self, query: str, top_k: int) -> list[dict]:
        """Score all chunks with BM25, return top-k."""
        bm25 = self._bm25_data["bm25"]
        chunks = self._bm25_data["chunks"]

        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)  # ndarray of shape (n_chunks,)

        # Pair each chunk with its score and sort descending
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [
            {
                "text": chunks[idx]["text"],
                "title": chunks[idx]["title"],
                "source_url": chunks[idx]["source_url"],
                "doc_type": chunks[idx]["doc_type"],
                "node_id": chunks[idx]["node_id"],
                "score": float(score),
            }
            for idx, score in ranked
            if score > 0  # skip zero-score results
        ]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_merge(
        dense_results: list[dict],
        sparse_results: list[dict],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.
        RRF score = sum over lists of 1 / (k + rank)
        where rank is 1-based position in the list.
        """
        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        chunk_store: dict[str, dict] = {}

        for rank, chunk in enumerate(dense_results, start=1):
            key = chunk["node_id"] or chunk["text"][:80]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            dense_ranks[key] = rank
            chunk_store[key] = chunk

        for rank, chunk in enumerate(sparse_results, start=1):
            key = chunk["node_id"] or chunk["text"][:80]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            sparse_ranks[key] = rank
            if key not in chunk_store:
                chunk_store[key] = chunk

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        return [
            RetrievedChunk(
                text=chunk_store[key]["text"],
                title=chunk_store[key]["title"],
                source_url=chunk_store[key]["source_url"],
                doc_type=chunk_store[key]["doc_type"],
                node_id=chunk_store[key]["node_id"],
                rrf_score=rrf_scores[key],
                dense_rank=dense_ranks.get(key),
                sparse_rank=sparse_ranks.get(key),
            )
            for key in sorted_keys[:top_k]
            if (chunk := chunk_store[key]) is not None
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        hyde_embedding: list[float] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Run hybrid retrieval for a query.

        Args:
            query:           Natural-language NPS question.
            top_k:           Number of chunks to return after RRF merge.
            hyde_embedding:  Optional pre-computed HyDE vector. When provided,
                             the dense leg uses this embedding instead of
                             re-encoding the raw query. The sparse leg always
                             uses the original query text.

        Returns:
            List of RetrievedChunk sorted by RRF score descending.
        """
        dense_results = self._dense_search(
            query, top_k=DENSE_FETCH, precomputed_embedding=hyde_embedding
        )
        sparse_results = self._sparse_search(query, top_k=SPARSE_FETCH)

        merged = self._rrf_merge(dense_results, sparse_results, top_k=top_k)
        return merged
