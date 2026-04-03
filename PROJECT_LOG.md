# NPS Information Chatbot — Project Log

**Project:** RAG-based chatbot for India's National Pension System (NPS)
**Student:** Shubh Sudan (ss2401@rit.edu) — Rochester Institute of Technology, NLP Course
**Live URL:** https://nps-chatbot.onrender.com
**GitHub:** https://github.com/shubhsudan/NPS-Chatbot

---

## What We Built

A production-deployed, end-to-end RAG (Retrieval-Augmented Generation) chatbot that answers natural language questions about India's National Pension System. Users ask questions about NPS rules — withdrawals, tax benefits, PRAN registration, annuity, Tier I/II accounts — and the bot returns accurate, grounded answers with source citations from official government documents.

---

## Architecture

```
User Query
    │
    ▼
PII Sanitizer ──── strips PRAN, Aadhaar, bank account numbers
    │
    ▼
Topic Guard ─────── LLM classifier: blocks clearly off-topic queries
    │
    ▼
HyDE ────────────── Groq generates hypothetical answer → embed it
    │
    ▼
Hybrid Retrieval ── Dense (Qdrant vector search) + Sparse (BM25) → RRF merge
    │
    ▼
Cross-Encoder ───── ms-marco-MiniLM-L-6-v2 reranks top candidates
    │
    ▼
Groq LLM ────────── llama-3.1-8b-instant generates grounded answer
    │
    ▼
FastAPI + Frontend ─ Returns answer + source citations to user
```

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Scraping | requests + BeautifulSoup + pdfplumber | Polite scraping with 1.5s delay |
| Chunking | LlamaIndex HierarchicalNodeParser | 2048 → 512 → 128 chunk sizes |
| Embeddings (local) | bge-large-en-v1.5 | 1024-dim, strong on regulatory text |
| Embeddings (Render) | bge-small-en-v1.5 | 384-dim, fits free tier memory |
| Vector DB | Qdrant Cloud (free tier) | Persistent across Render restarts |
| Sparse retrieval | rank_bm25 | Handles exact keyword queries |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 | Big accuracy boost |
| LLM | Groq API — llama-3.1-8b-instant | Free tier, 500k TPD |
| Backend | FastAPI | Lazy pipeline loading for fast startup |
| Frontend | Plain HTML/CSS/JS | No build step, served from FastAPI |
| Hosting | Render (free tier) | Web service, auto-deploy from GitHub |
| Python env | conda, Python 3.11 | 3.13 caused segfaults with PyTorch |

---

## Phase 1 — Ingestion Pipeline

### Files
- `ingestion/scraper.py` — scrapes HTML pages and downloads PDFs
- `ingestion/chunker.py` — hierarchical chunking via LlamaIndex
- `ingestion/embedder.py` — embeds chunks, reads `EMBED_MODEL` from env
- `ingestion/upload_to_qdrant.py` — upserts to Qdrant + builds BM25 index
- `ingestion/run_ingestion.py` — orchestrates all 4 steps; supports `--env` flag

### Running ingestion

**Local (bge-large → nps_docs):**
```bash
cd ingestion
python run_ingestion.py
```

**For Render (bge-small → nps_docs_small):**
```bash
cd ingestion
python run_ingestion.py --env .env.small
```

### Data sources (170+ official documents)

**HTML pages (18):** npstrust.org.in — FAQs, about NPS, eligibility, tax benefits, charges, withdrawals (normal/premature/partial/death/deferment), open account, Tier II, NPS Vatsalya, APY, circulars listing, annual reports, tenders, RTI

**FAQ PDFs (12):** Central Govt sector, State Govt sector, All Citizens model, Exit FAQs (CG/SG/All Citizens/Corporate), NRI, eNPS, Retirement Adviser, NPS Ombudsman, Partial Withdrawal, APY, NPS Vatsalya, UPS

**Regulation PDFs (51):** Full history of PFRDA Acts and Regulations 2015–2025 including:
- PFRDA Act 2013 (primary legislation)
- Exit & Withdrawal Regulations (base + all 5 amendments)
- Pension Fund Regulations (base + 6 amendments)
- NPS Trust Regulations (base + all amendments)
- CRA Regulations (base + all amendments)
- POP Regulations (base + all amendments)
- Custodian, Trustee Bank, Aggregator, Subscriber Grievance Regulations
- APY eligibility amendment, UPS operationalisation (2025)

**Withdrawal PDFs (5):** Exit FAQs for CG, SG, All Citizens, Corporate, Partial Withdrawal

**Annual Reports (2):** 2022-23, 2023-24

**RTI Disclosures (4):** Norms, Rules/Regulations/Manuals, Categories of Documents, FY2023-24 disclosures

**Local PDFs (25):** Official FAQ PDFs manually saved to `data/local_docs/`

**PFRDA listing pages (2):** Active circulars and guidelines pages

### Ingestion results

| Run | Model | Chunks | Collection |
|-----|-------|--------|------------|
| Run 1 (local) | bge-large-en-v1.5 | 5,564 | nps_docs |
| Run 2 (Render) | bge-small-en-v1.5 | 11,729 | nps_docs_small |

---

## Phase 2 — Retrieval

### Files
- `backend/pipeline/retriever.py` — HybridRetriever class

### How it works
1. **Dense search** — embed query with bge (or use HyDE vector) → Qdrant ANN search (top 20)
2. **Sparse search** — BM25 over all chunk texts (top 20); skipped gracefully if index missing
3. **RRF merge** — Reciprocal Rank Fusion (k=60) combines both lists → returns top 10

---

## Phase 3 — Full Pipeline

### Files
- `backend/pipeline/sanitizer.py` — regex strips PRAN (12-digit), Aadhaar, bank accounts, UPI, IFSC
- `backend/pipeline/topic_guard.py` — Groq LLM classifier; generous scope (NPS/pension/retirement/tax); fails open (YES when in doubt)
- `backend/pipeline/hyde.py` — Groq generates hypothetical answer → bge embeds it → returns vector
- `backend/pipeline/reranker.py` — CrossEncoder ms-marco-MiniLM-L-6-v2 reranks top 10 → top 4
- `backend/pipeline/generator.py` — locked NPS-only system prompt, Groq API call, citation deduplication
- `backend/pipeline/pipeline.py` — orchestrates all steps; returns PipelineResult with answer + citations

---

## Phase 4 — FastAPI Backend

### Files
- `backend/main.py` — FastAPI app

### Endpoints
- `POST /chat` — main chat endpoint; lazy-loads pipeline on first call
- `GET /health` — returns `{"status": "ok"}` immediately (no model dependency)

### Key design decisions
- **Lazy loading** — pipeline loads on first `/chat` request, not at startup; lets Render bind port immediately
- **StaticFiles** — frontend served from FastAPI on same port; no CORS needed
- **Rate limit handling** — `GroqRateLimitError` caught → HTTP 429 with friendly message
- **PII** — sanitized before any LLM call; UI shows notice if query was modified

---

## Phase 5 — Frontend + Deploy

### Files
- `frontend/index.html` — chat UI with disclaimer, suggestion buttons
- `frontend/style.css` — blue/white gov-service styling, responsive
- `frontend/app.js` — calls `/chat`, renders answer with markdown formatting + citations

### Render deployment
- **URL:** https://nps-chatbot.onrender.com
- **Service:** Web Service, Free tier (512 MB / 0.1 CPU)
- **Build command:** `pip install -r backend/requirements.txt && python -c "...pre-download models..."`
- **Start command:** `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Auto-deploy:** pushes to `main` branch trigger automatic redeploy

### Environment variables on Render
| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | from console.groq.com |
| `QDRANT_URL` | from cloud.qdrant.io |
| `QDRANT_API_KEY` | from cloud.qdrant.io |
| `QDRANT_COLLECTION` | `nps_docs_small` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |

---

## Phase 6 — Evaluation

### Files
- `evaluate/evaluate.py` — RAGAS evaluation framework
- `evaluate/test_questions.py` — 25 gold-standard NPS Q&A pairs

### Metrics
- **Faithfulness** — does the answer stay grounded in retrieved chunks?
- **Answer Relevancy** — does the answer address the question?
- **Context Precision** — are the retrieved chunks actually relevant?
- **Context Recall** — are all relevant chunks being retrieved?

### Running evaluation
```bash
cd evaluate
python evaluate.py
```

---

## Bugs Fixed

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Segfault on startup | Python 3.13 incompatible with PyTorch | Created conda env with Python 3.11 |
| 404 on NPS Trust URLs | Website restructured | Updated all 18 HTML source URLs |
| `TypeError: Document not subscriptable` | Scraper returns dataclasses, chunker expects dicts | `[asdict(d) for d in run_scraper()]` in run_ingestion.py |
| Qdrant 404 | Used node-level URL with `node-0-` prefix | Removed prefix from QDRANT_URL |
| `httpx proxies` error | groq package incompatible with httpx 0.28 | `pip install "httpx<0.28"` |
| `[object Object]` in UI | Pydantic validation errors return `detail` as array | `Array.isArray(err.detail) ? err.detail.map(e=>e.msg).join(", ") : err.detail` |
| CORS / Load failed | Frontend on port 5500, backend on 8000 | Serve frontend from FastAPI StaticFiles on single port |
| Server error 500 | Groq daily 100k token limit | Added GroqRateLimitError → HTTP 429; switched to llama-3.1-8b-instant (500k TPD) |
| Render: `requirements.txt` not found | Build command ran from root, file is at `backend/requirements.txt` | Fixed build command path |
| Render: `No open ports detected` | Models loading at startup → uvicorn never bound port | Lazy-load pipeline on first `/chat` request |
| Render: `KeyError: QDRANT_API_KEY` | Env vars not saved on Render | Added all 6 env vars via Render dashboard |
| 404 on circular PDFs | NPS Trust circulars require JS session | Removed from DIRECT_PDF_SOURCES; page text still scraped |

---

## Local Development

### Setup
```bash
conda create -n nps-chatbot python=3.11
conda activate nps-chatbot
pip install -r ingestion/requirements.txt
pip install -r backend/requirements.txt
```

### Run locally
```bash
# One command:
./start.sh

# Or manually:
cd ingestion && python run_ingestion.py   # only needed once
cd backend && uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

### Environment (.env at project root)
```
GROQ_API_KEY=...
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION=nps_docs
EMBED_MODEL=BAAI/bge-large-en-v1.5
GROQ_MODEL=llama-3.1-8b-instant
```

---

## Folder Structure

```
nps-chatbot/
├── ingestion/
│   ├── scraper.py              # 170+ source documents (HTML + PDFs)
│   ├── chunker.py              # LlamaIndex hierarchical chunking
│   ├── embedder.py             # bge embeddings (env-configurable)
│   ├── upload_to_qdrant.py     # Qdrant upsert + BM25 index build
│   └── run_ingestion.py        # Full pipeline orchestrator (--env flag)
│
├── backend/
│   ├── main.py                 # FastAPI app (lazy pipeline, StaticFiles)
│   ├── pipeline/
│   │   ├── sanitizer.py        # PII scrubbing
│   │   ├── topic_guard.py      # LLM-based topic classifier
│   │   ├── hyde.py             # Hypothetical Document Embeddings
│   │   ├── retriever.py        # Hybrid dense+sparse retrieval (RRF)
│   │   ├── reranker.py         # Cross-encoder reranking
│   │   └── generator.py        # Groq LLM answer generation
│   └── requirements.txt
│
├── frontend/
│   ├── index.html              # Chat UI
│   ├── style.css               # Styling
│   └── app.js                  # API calls + markdown rendering
│
├── evaluate/
│   ├── evaluate.py             # RAGAS evaluation
│   └── test_questions.py       # 25 gold-standard Q&A pairs
│
├── data/
│   ├── local_docs/             # 25 manually saved PDFs (committed)
│   ├── raw/                    # Scraped documents (gitignored)
│   └── processed/              # Chunks + embeddings (gitignored, BM25 committed)
│
├── .env                        # Secrets (gitignored)
├── .env.example                # Template (committed)
├── .env.small                  # Render config: bge-small + nps_docs_small (gitignored)
├── render.yaml                 # Render deployment config
├── start.sh                    # One-command local startup
└── CLAUDE.md                   # Project context for Claude Code
```

---

## Key Papers Referenced

1. Lewis et al. (2020) — Original RAG paper — foundational architecture
2. Karna et al. (2026) — Hybrid RAG-LLaMA for Indian legal texts — closest blueprint
3. Gholap et al. (2026) — Integrated RAG for Indian tax law — FAISS + LLaMA-3 approach
4. Choi et al. (2024) — LLMs as tax attorneys — evaluation methodology
5. RAGAS (2023) — Evaluation framework for RAG systems
