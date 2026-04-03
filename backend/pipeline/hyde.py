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

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

_groq_client: Groq | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client


# ---------------------------------------------------------------------------
# HyDE
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = (
    "You are a knowledgeable assistant specialising in India's National Pension System (NPS). "
    "When given a question, write a short, factual answer (3-5 sentences) as if it appears in "
    "an official NPS document or FAQ. Be specific and use the terminology found in PFRDA regulations."
)


def generate_hyde_text(query: str) -> str:
    """
    Ask Groq to write a short hypothetical NPS answer to the query.
    Returns the hypothetical text (or the original query on failure).
    The caller is responsible for embedding it — no SentenceTransformer loaded here.
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
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[HyDE] Groq call failed ({e}), falling back to raw query.")
        return query
