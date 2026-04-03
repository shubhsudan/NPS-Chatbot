# NPS Information Chatbot — Project Context for Claude Code

## What This Project Is

A RAG-based chatbot that helps Indian citizens navigate the National Pension System (NPS),
governed by PFRDA (Pension Fund Regulatory and Development Authority). Users ask natural
language questions about NPS rules — withdrawals, annuity, tax treatment, tier I/II accounts,
PRAN registration — and the bot returns accurate, step-by-step guidance grounded in official
government documents. The target user has no financial or legal background.

This is an academic NLP project for Rochester Institute of Technology (RIT).
Student: Shubh Sudan (ss2401@rit.edu)

---

## Architecture Overview

The system has three layers:

### 1. Offline Ingestion Pipeline (run once + weekly refresh)
- Scrape official NPS documents from npstrust.org.in and pfrda.org.in
- Parse HTML pages and PDFs
- Chunk documents using LlamaIndex hierarchical node parser (preserves section headers)
- Embed chunks using bge-large-en-v1.5 (HuggingFace sentence-transformers)
- Store dense vectors in Qdrant Cloud (free tier)
- Build and serialize a BM25 index for sparse/keyword retrieval

### 2. FastAPI Backend (deployed on Render)
Query pipeline for every user message:
1. PII Sanitizer — strip PRAN numbers, bank account patterns before processing
2. HyDE — generate a hypothetical answer to the query, embed it for better retrieval
3. Hybrid Retrieval — dense search (Qdrant) + sparse search (BM25), merge results
4. Cross-Encoder Re-ranking — re-score retrieved chunks with ms-marco cross-encoder
5. Groq API (Llama 3.1 70B) — generate final response with strict system prompt
6. Return response + source citations to frontend

Endpoints:
- POST /chat — main chat endpoint
- GET /health — health check to prevent Render free tier cold starts

### 3. Frontend (Render Static Site)
- Plain HTML/CSS/JS (no framework, no build step)
- Chat interface with source citation display
- Disclaimer banner: "Do not enter your PRAN, bank account, or personal IDs"
- Calls /chat endpoint, renders streamed or JSON response

---

## Tech Stack — Final Decisions

| Layer | Tool | Notes |
|---|---|---|
| Scraping | requests + BeautifulSoup + pdfplumber | Free, no API |
| Chunking | LlamaIndex hierarchical node parser | Keeps section headers with chunks |
| Embeddings | bge-large-en-v1.5 (sentence-transformers) | Free, strong on formal text |
| Vector DB | Qdrant Cloud free tier | Persistent across Render restarts |
| Sparse retrieval | rank_bm25 | Handles exact keyword queries |
| Re-ranking | cross-encoder/ms-marco (HuggingFace) | Free, big accuracy boost |
| LLM | Groq API — Llama 3.1 70B | Free tier, fast, no GPU needed |
| Backend | FastAPI | Async, lightweight, easy on Render |
| Frontend | Plain HTML/CSS/JS | No build step |
| Hosting | Render | Free tier (web service + static site) |

---

## Project Folder Structure

```
nps-chatbot/
│
├── ingestion/
│   ├── scraper.py              # Scrapes HTML + downloads PDFs from NPS/PFRDA sites
│   ├── chunker.py              # Hierarchical chunking via LlamaIndex
│   ├── embedder.py             # Embeds chunks with bge-large-en-v1.5
│   └── upload_to_qdrant.py     # Pushes embedded chunks to Qdrant Cloud
│
├── backend/
│   ├── main.py                 # FastAPI app — /chat and /health
│   ├── pipeline/
│   │   ├── sanitizer.py        # PII scrubbing (PRAN regex, account number patterns)
│   │   ├── hyde.py             # HyDE query expansion
│   │   ├── retriever.py        # Hybrid dense (Qdrant) + sparse (BM25) retrieval
│   │   ├── reranker.py         # Cross-encoder re-ranking
│   │   └── generator.py        # Groq API call + system prompt
│   └── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js                  # Calls /chat, renders response + citations
│
├── .env                        # GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
├── .env.example                # Safe to commit — template with empty values
├── render.yaml                 # Render deployment config
└── CLAUDE.md                   # This file — read at the start of every session
```

---

## Data Sources

Scrape these URLs for official NPS documents:
- https://www.npstrust.org.in/content/faq (FAQs)
- https://www.npstrust.org.in/content/withdrawal (Withdrawal rules)
- https://www.pfrda.org.in/index1.cshtml?lngId=25 (PFRDA circulars)
- https://www.pfrda.org.in/index1.cshtml?lngId=6 (Guidelines)
- https://enps.nsdl.com/eNPS/NationalPensionSystem.html (CRA operational details)

All are public government websites — no authentication, no scraping restrictions.
Schedule a weekly refresh using a cron job or Python APScheduler.

---

## Security Requirements

- Stateless by design — no conversation history written to disk or database
- PII sanitizer must strip: 12-digit PRAN numbers, bank account numbers, Aadhaar patterns
- System prompt must be locked — user input cannot override the bot's role
- Add explicit UI disclaimer: users should not enter personal identifiers
- HTTPS enforced by Render (automatic)
- No third-party analytics or logging of user messages

---

## Build Order (Phases)

Build in this exact order — do not skip ahead:

**Phase 1 — Ingestion**
Build scraper.py, chunker.py, embedder.py, upload_to_qdrant.py
Goal: NPS documents chunked and loaded into Qdrant Cloud

**Phase 2 — Retrieval only**
Build retriever.py with hybrid dense + BM25 search
Test: given a real NPS question, confirm relevant chunks come back
Do NOT connect the LLM yet

**Phase 3 — Full pipeline**
Add sanitizer.py, hyde.py, reranker.py, generator.py
Test end-to-end in a standalone Python script first

**Phase 4 — FastAPI**
Wrap pipeline in main.py with /chat and /health endpoints
Test locally with curl or Postman

**Phase 5 — Frontend + Deploy**
Build index.html, style.css, app.js
Write render.yaml
Deploy to Render

**Phase 6 — Evaluate**
Use RAGAS framework (faithfulness, answer relevance, context precision)
Test on 20-30 real NPS questions covering withdrawals, annuity, tax, registration

---

## Key Papers Referenced (for context)

1. Lewis et al. (2020) — Original RAG paper — foundational architecture
2. Karna et al. (2026) — Hybrid RAG-LLaMA for Indian legal texts — closest blueprint
3. Gholap et al. (2026) — Integrated RAG for Indian tax law — FAISS + LLaMA-3 approach
4. Choi et al. (2024) — LLMs as tax attorneys — evaluation methodology (4 retrieval methods)
5. RAGAS (2023) — Evaluation framework for RAG systems

---

## Environment Variables Required

```
GROQ_API_KEY=          # From console.groq.com (free, no credit card)
QDRANT_URL=            # From cloud.qdrant.io (free tier)
QDRANT_API_KEY=        # From cloud.qdrant.io
QDRANT_COLLECTION=nps_docs
EMBED_MODEL=BAAI/bge-large-en-v1.5
GROQ_MODEL=llama-3.1-70b-versatile
```

---

## Current Status

Project is in planning/setup phase. No code written yet.
Start with Phase 1 — build ingestion/scraper.py first.
