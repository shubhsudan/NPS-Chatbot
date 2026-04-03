"""
hyde.py — Hypothetical Document Embeddings (HyDE) for query expansion.

HyDE (Gao et al. 2022):
  1. Ask the LLM to write a short hypothetical answer to the query.
  2. Embed that hypothetical answer instead of the raw query.
  3. Use the resulting vector for dense retrieval.

Why this helps: the hypothetical answer lives in the same vector space as
real document chunks, so cosine search finds closer neighbours than a terse
question would.

The sparse (BM25) leg always uses the original query — HyDE only improves
the dense leg.
"""

import os

import torch
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_groq_client: Groq | None = None
_embed_model: SentenceTransformer | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    return _embed_model


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = (
    "You are a knowledgeable assistant specialising in India's National Pension System (NPS). "
    "When given a question, write a short, factual answer (3-5 sentences) as if it appears in "
    "an official NPS document or FAQ. Be specific and use the terminology found in PFRDA regulations."
)


def generate_hyde_embedding(query: str) -> list[float]:
    """
    Generate a HyDE embedding for the given query.

    Returns a normalised 1024-dim vector (list[float]) ready for Qdrant search.
    Falls back to embedding the original query if the Groq call fails.
    """
    try:
        response = _get_groq().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        hypothetical_text = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[HyDE] Groq call failed ({e}), falling back to raw query embedding.")
        hypothetical_text = query

    model = _get_embed_model()
    vec = model.encode(
        BGE_QUERY_PREFIX + hypothetical_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vec.tolist()
