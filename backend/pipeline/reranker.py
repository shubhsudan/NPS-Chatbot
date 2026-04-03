"""
reranker.py — Cross-encoder re-ranking of retrieved chunks.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 (HuggingFace, free).
The cross-encoder jointly encodes (query, passage) pairs and produces
a relevance score — more accurate than bi-encoder cosine similarity
but too slow to run over the full corpus, so we run it only on the
small candidate set returned by the hybrid retriever.
"""

from sentence_transformers import CrossEncoder
from pipeline.retriever import RetrievedChunk
from dataclasses import dataclass

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        print(f"[Reranker] Loading cross-encoder: {CROSS_ENCODER_MODEL}")
        _model = CrossEncoder(CROSS_ENCODER_MODEL, max_length=512)
    return _model


@dataclass
class RankedChunk:
    text: str
    title: str
    source_url: str
    doc_type: str
    node_id: str
    ce_score: float          # cross-encoder logit score (higher = more relevant)
    rrf_score: float         # original RRF score kept for transparency


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RankedChunk]:
    """
    Re-score chunks with the cross-encoder and return sorted by ce_score.

    Args:
        query:  Original (sanitized) user query.
        chunks: Candidate chunks from the hybrid retriever.
        top_k:  If set, return only the top-k results after re-ranking.
                If None, return all chunks in re-ranked order.

    Returns:
        List of RankedChunk sorted by ce_score descending.
    """
    if not chunks:
        return []

    model = _get_model()

    pairs = [(query, chunk.text) for chunk in chunks]
    scores: list[float] = model.predict(pairs).tolist()

    ranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    if top_k is not None:
        ranked = ranked[:top_k]

    return [
        RankedChunk(
            text=chunk.text,
            title=chunk.title,
            source_url=chunk.source_url,
            doc_type=chunk.doc_type,
            node_id=chunk.node_id,
            ce_score=score,
            rrf_score=chunk.rrf_score,
        )
        for chunk, score in ranked
    ]
