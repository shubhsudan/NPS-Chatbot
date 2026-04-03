"""
test_retrieval.py — Phase 2 smoke test.

Runs the HybridRetriever against a set of real NPS questions and
prints the top-k results so you can visually verify relevance.
No LLM, no FastAPI — pure retrieval only.

Run from the backend/ directory:
    python test_retrieval.py
"""

import textwrap
from pipeline.retriever import HybridRetriever

# Representative NPS questions covering the main topic areas
TEST_QUERIES = [
    "Can I withdraw my entire NPS corpus before retirement at 60?",
    "What is the minimum annuity purchase requirement on exit from NPS?",
    "How is NPS contribution taxed under Section 80CCD?",
    "What is the difference between NPS Tier I and Tier II accounts?",
    "How do I register for PRAN online through eNPS?",
    "What happens to NPS if a subscriber dies before 60?",
    "Can a government employee make voluntary contributions to NPS?",
    "What is the partial withdrawal rule for NPS Tier I?",
]

TOP_K = 5


def print_separator(char: str = "-", width: int = 70) -> None:
    print(char * width)


def run_tests() -> None:
    retriever = HybridRetriever()

    for i, query in enumerate(TEST_QUERIES, start=1):
        print_separator("=")
        print(f"Query {i}/{len(TEST_QUERIES)}: {query}")
        print_separator()

        results = retriever.retrieve(query, top_k=TOP_K)

        if not results:
            print("  [!] No results returned — check that Qdrant has data.")
            continue

        for rank, chunk in enumerate(results, start=1):
            print(f"\n  Rank {rank}  |  RRF: {chunk.rrf_score:.4f}"
                  f"  |  Dense rank: {chunk.dense_rank}"
                  f"  |  Sparse rank: {chunk.sparse_rank}")
            print(f"  Title:  {chunk.title}")
            print(f"  Source: {chunk.source_url}")
            # Show a 300-char snippet of the chunk text
            snippet = textwrap.fill(chunk.text[:300], width=66,
                                    initial_indent="  ", subsequent_indent="  ")
            print(f"  Text:\n{snippet}{'...' if len(chunk.text) > 300 else ''}")

        print()

    print_separator("=")
    print("Retrieval test complete.")


if __name__ == "__main__":
    run_tests()
