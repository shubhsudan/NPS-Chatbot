# Dockerfile — NPS Information Chatbot
# Targets Hugging Face Spaces (port 7860, Docker SDK)

FROM python:3.11-slim

# HF Spaces runs containers as a non-root user (UID 1000).
# Create the user early so file ownership is correct.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# System deps (pdfplumber / torch need these)
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------------------------
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Pre-download models at build time so startup is fast
# (models are cached in ~/.cache/huggingface inside the image)
# ---------------------------------------------------------------------------
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models pre-downloaded OK') \
"

# ---------------------------------------------------------------------------
# Copy application files
# ---------------------------------------------------------------------------
COPY backend/  ./backend/
COPY frontend/ ./frontend/
COPY data/     ./data/

# Fix ownership for non-root user
RUN chown -R appuser:appuser /app

USER appuser

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
EXPOSE 7860

WORKDIR /app/backend

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
