"""
pipeline.py — Orchestrates the full query pipeline.

Order:
  1. Sanitizer  — strip PII from user query
  2. HyDE       — generate hypothetical answer, embed it for better dense retrieval
  3. Retriever  — hybrid dense (HyDE embedding) + sparse (BM25) with RRF merge
  4. Reranker   — cross-encoder re-scores the candidate set
  5. Generator  — Groq LLM produces final grounded answer

Public API:
    pipeline = NPSPipeline()
    result   = pipeline.query("Can I withdraw NPS corpus before 60?")
    # result.answer    → str
    # result.citations → list[Citation]
"""

from dataclasses import dataclass, field

from pipeline.sanitizer import sanitize
from pipeline.topic_guard import is_in_scope
from pipeline.hyde import generate_hyde_embedding
from pipeline.retriever import HybridRetriever, RetrievedChunk
from pipeline.reranker import rerank, RankedChunk
from pipeline.generator import generate, Citation

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RETRIEVAL_TOP_K = 10   # candidates passed to the reranker
RERANK_TOP_K = 4       # final context chunks passed to the LLM


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    answer: str
    citations: list[Citation]
    sanitized_query: str
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    reranked_chunks: list[RankedChunk] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class NPSPipeline:
    """
    Load once, call query() many times. All heavy models are cached in memory.
    """

    def __init__(self) -> None:
        self._retriever = HybridRetriever()

    def query(self, user_query: str) -> PipelineResult:
        # 1. Sanitize PII
        clean_query = sanitize(user_query)

        # 2. Topic guard — only answer NPS-related questions
        if not is_in_scope(clean_query):
            return PipelineResult(
                answer=(
                    "I'm specifically designed to answer questions about India's "
                    "**National Pension System (NPS)**.\n\n"
                    "I can help with topics like:\n"
                    "- Withdrawals and exit rules\n"
                    "- Tax benefits (Section 80CCD)\n"
                    "- Tier I and Tier II accounts\n"
                    "- PRAN registration\n"
                    "- Annuity and retirement planning under NPS\n"
                    "- PFRDA regulations and circulars\n\n"
                    "Please ask an NPS-related question and I'll be happy to help."
                ),
                citations=[],
                sanitized_query=clean_query,
            )

        # 3. HyDE — generate hypothetical answer, embed for better dense retrieval
        hyde_vec = generate_hyde_embedding(clean_query)

        # 4. Hybrid retrieval using HyDE embedding for dense leg
        retrieved = self._retriever.retrieve(
            clean_query,
            top_k=RETRIEVAL_TOP_K,
            hyde_embedding=hyde_vec,
        )

        # 5. Cross-encoder re-ranking
        reranked = rerank(clean_query, retrieved, top_k=RERANK_TOP_K)

        # 6. Generate grounded answer with NPS context
        answer, citations = generate(clean_query, reranked)

        return PipelineResult(
            answer=answer,
            citations=citations,
            sanitized_query=clean_query,
            retrieved_chunks=retrieved,
            reranked_chunks=reranked,
        )
