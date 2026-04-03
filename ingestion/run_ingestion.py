"""
run_ingestion.py — Runs the full Phase 1 ingestion pipeline end-to-end.

Steps:
  1. scraper.py  — fetch HTML + PDFs → data/raw/documents.json
  2. chunker.py  — hierarchical chunking → data/processed/leaf_chunks.json
  3. embedder.py — embed leaf chunks   → data/processed/embedded_chunks.json
  4. upload      — push to Qdrant + build BM25 index

Run from the project root:
  python ingestion/run_ingestion.py                    # uses .env (bge-large, nps_docs)
  python ingestion/run_ingestion.py --env .env.small   # uses .env.small (bge-small, nps_docs_small)
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Parse optional --env flag before importing modules that read env vars
_env_file = ".env"
if "--env" in sys.argv:
    idx = sys.argv.index("--env")
    if idx + 1 < len(sys.argv):
        _env_file = sys.argv[idx + 1]

load_dotenv(Path(__file__).parent.parent / _env_file, override=True)
print(f"Loaded env from: {_env_file}")

from dataclasses import asdict
from scraper import run_scraper
from chunker import run_chunker
from embedder import run_embedder
from upload_to_qdrant import run_upload

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1 — NPS Ingestion Pipeline")
    print("=" * 60)

    print("\n[1/4] Scraping documents...")
    documents = [asdict(d) for d in run_scraper()]

    print("\n[2/4] Chunking documents...")
    leaf_chunks = run_chunker(documents)

    print("\n[3/4] Embedding chunks...")
    embedded_chunks = run_embedder(leaf_chunks)

    print("\n[4/4] Uploading to Qdrant + building BM25 index...")
    run_upload(embedded_chunks, recreate=True)

    print("\n✓ Phase 1 complete.")
