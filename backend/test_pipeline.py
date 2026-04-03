"""
test_pipeline.py — Phase 3 end-to-end test.

Runs the full pipeline (sanitizer → HyDE → retrieval → reranker → generator)
on a set of real NPS questions and prints answers + citations.
No FastAPI involved — raw Python script.

Run from the backend/ directory:
    python test_pipeline.py
"""

import textwrap
from pipeline.pipeline import NPSPipeline

TEST_QUERIES = [
    "Can I withdraw my entire NPS corpus before retirement at age 60?",
    "What is the minimum annuity requirement when I exit NPS at 60?",
    "How is NPS contribution taxed — what is the Section 80CCD limit?",
    "What is the difference between NPS Tier I and Tier II accounts?",
    "How do I open a PRAN account online through eNPS?",
    "What happens to my NPS account if I die before retirement?",
    "Can I make partial withdrawals from NPS Tier I before 60?",
]

# PII injection test — sanitizer should strip these before they reach the LLM
PII_TEST_QUERIES = [
    "My PRAN is 110098765432. Can I withdraw before 60?",
    "My Aadhaar is 9876 5432 1098. How do I update my NPS details?",
]


def print_sep(char: str = "-", width: int = 72) -> None:
    print(char * width)


def run() -> None:
    print("Loading pipeline (this may take a minute on first run)...")
    pipeline = NPSPipeline()

    # --- Normal queries ---
    for i, q in enumerate(TEST_QUERIES, start=1):
        print_sep("=")
        print(f"Q{i}: {q}")
        print_sep()

        result = pipeline.query(q)

        print(f"Sanitized:  {result.sanitized_query}")
        print(f"Retrieved:  {len(result.retrieved_chunks)} chunks  |  "
              f"Reranked: {len(result.reranked_chunks)} chunks")
        print()
        print("ANSWER:")
        print(textwrap.fill(result.answer, width=72))
        print()
        print("SOURCES:")
        for c in result.citations:
            print(f"  • {c.title}")
            print(f"    {c.source_url}")
        print()

    # --- PII sanitizer test ---
    print_sep("=")
    print("PII SANITIZATION TEST")
    print_sep()
    for q in PII_TEST_QUERIES:
        result = pipeline.query(q)
        print(f"Original:  {q}")
        print(f"Sanitized: {result.sanitized_query}")
        pii_leaked = any(
            identifier in result.answer
            for identifier in ["110098765432", "9876 5432 1098", "9876543210198"]
        )
        print(f"PII in answer: {'FAIL — PII leaked!' if pii_leaked else 'OK'}")
        print()

    print_sep("=")
    print("Phase 3 test complete.")


if __name__ == "__main__":
    run()
