---
title: NPS Information Chatbot
emoji: 🏦
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# NPS Information Chatbot

A RAG-based chatbot that answers questions about India's National Pension System (NPS), grounded in official PFRDA documents.

**Live:** https://huggingface.co/spaces/shubhsudan/nps-chatbot

## What it does

Users ask natural-language questions about NPS rules — withdrawals, tax benefits, PRAN registration, annuity, Tier I/II accounts — and the bot returns accurate, step-by-step answers with source citations from official government documents.

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
HyDE ────────────── Groq generates hypothetical answer text
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

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | Qdrant Cloud |
| Sparse retrieval | BM25 (rank_bm25) |
| Re-ranking | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM | Groq API — llama-3.1-8b-instant |
| Backend | FastAPI |
| Frontend | Plain HTML/CSS/JS |

## Academic context

RIT NLP Course project — Shubh Sudan (ss2401@rit.edu)
Rochester Institute of Technology, 2026
