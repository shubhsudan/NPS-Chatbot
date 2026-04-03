#!/bin/bash
# start.sh — One command to run the NPS chatbot
# Usage: bash start.sh
#        bash start.sh --reingest   (re-scrape and re-upload documents)

ROOT="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="nps-chatbot"

# ── Activate conda env ────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo ""
echo "============================================"
echo "  NPS Chatbot"
echo "============================================"
echo ""

# ── Optional: re-run ingestion ────────────────────────────────────
if [[ "$1" == "--reingest" ]]; then
    echo "Running ingestion pipeline (this takes ~10 min)..."
    cd "$ROOT/ingestion"
    python run_ingestion.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Ingestion failed. Check output above."
        exit 1
    fi
    echo "Ingestion complete."
    echo ""
fi

# ── Check ingestion has been run ──────────────────────────────────
if [ ! -f "$ROOT/data/processed/bm25_index.pkl" ]; then
    echo "ERROR: Ingestion has not been run yet."
    echo "Run this first:  bash start.sh --reingest"
    exit 1
fi

# ── Kill anything already on port 8000 ───────────────────────────
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:5500 | xargs kill -9 2>/dev/null
sleep 1

# ── Start single server (backend serves frontend too) ─────────────
echo "Starting server..."
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 > "$ROOT/backend.log" 2>&1 &
SERVER_PID=$!

echo "Waiting for models to load (may take ~30s)..."
for i in {1..40}; do
    sleep 2
    if curl -s http://localhost:8000/health 2>/dev/null | grep -q "ok"; then
        echo "Ready."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Server crashed. Check backend.log"
        cat "$ROOT/backend.log"
        exit 1
    fi
done

# ── Open browser ──────────────────────────────────────────────────
open "http://localhost:8000" 2>/dev/null || true

echo ""
echo "============================================"
echo "  NPS Chatbot is running"
echo "--------------------------------------------"
echo "  Chat UI  →  http://localhost:8000"
echo "  API docs →  http://localhost:8000/docs"
echo "--------------------------------------------"
echo "  Logs     →  tail -f backend.log"
echo "  Stop     →  Ctrl+C"
echo "============================================"
echo ""

trap "echo ''; echo 'Shutting down...'; kill $SERVER_PID 2>/dev/null; exit 0" INT
wait
