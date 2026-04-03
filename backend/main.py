"""
main.py — FastAPI backend for the NPS chatbot.

Endpoints:
  POST /chat    — run the full pipeline, return answer + citations
  GET  /health  — liveness check (also used to prevent Render cold starts)

Run locally:
  uvicorn main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from groq import RateLimitError as GroqRateLimitError
from pydantic import BaseModel, Field

from pipeline.pipeline import NPSPipeline, PipelineResult

load_dotenv()

# ---------------------------------------------------------------------------
# Lifespan — load heavy models once at startup
# ---------------------------------------------------------------------------

_pipeline: NPSPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("[startup] Loading NPS pipeline...")
    _pipeline = NPSPipeline()
    print("[startup] Pipeline ready.")
    yield
    _pipeline = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NPS Chatbot API",
    description="RAG-based chatbot for India's National Pension System (NPS). "
                "Academic project — RIT / Shubh Sudan.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the Render static-site frontend (and localhost dev) to call the API
_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1",
    "http://127.0.0.1:5500",
    os.environ.get("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in _ALLOWED_ORIGINS if o],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The user's NPS question.",
        examples=["Can I withdraw NPS corpus before retirement at 60?"],
    )


class CitationResponse(BaseModel):
    title: str
    source_url: str
    doc_type: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    sanitized_query: str   # returned so the UI can show a note if PII was removed


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict:
    """
    Liveness check. Returns 200 when the pipeline is loaded and ready.
    Ping this endpoint every 5 minutes from an external cron to prevent
    Render free-tier cold starts.
    """
    ready = _pipeline is not None
    return {"status": "ok" if ready else "warming_up", "pipeline_loaded": ready}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the full RAG pipeline and return an answer grounded in official NPS documents.

    The query is sanitized for PII before any processing. The response includes
    source citations so the user can verify the information.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline is still loading. Try again shortly.")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        result: PipelineResult = _pipeline.query(request.query)
    except GroqRateLimitError:
        raise HTTPException(
            status_code=429,
            detail="The AI service is temporarily rate-limited. Please wait a few minutes and try again."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    return ChatResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                title=c.title,
                source_url=c.source_url,
                doc_type=c.doc_type,
            )
            for c in result.citations
        ],
        sanitized_query=result.sanitized_query,
    )


# ---------------------------------------------------------------------------
# Serve frontend — must be mounted AFTER API routes
# ---------------------------------------------------------------------------

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(_FRONTEND_DIR / "index.html")

app.mount("/", StaticFiles(directory=_FRONTEND_DIR), name="frontend")
