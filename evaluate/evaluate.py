"""
evaluate.py — Phase 6: RAGAS evaluation of the NPS RAG pipeline.

Runs the full pipeline on 25 gold-standard NPS questions and measures:
  - Faithfulness       : does the answer stay grounded in the retrieved context?
  - Answer Relevancy   : is the answer actually relevant to the question?
  - Context Precision  : are the retrieved chunks relevant to the question?
  - Context Recall     : do the retrieved chunks cover the ground-truth answer?

Results are printed as a table and saved to evaluate/results/ragas_results.csv
and evaluate/results/per_question.json for inspection.

Usage (from project root):
    cd evaluate
    python evaluate.py

    # Evaluate only a subset:
    python evaluate.py --limit 5

    # Skip expensive LLM metrics, only run context recall:
    python evaluate.py --metrics context_recall
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Add backend to path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

load_dotenv()

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RAGAS evaluation for NPS chatbot")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only the first N questions (default: all 25)")
    p.add_argument("--metrics", type=str, default="all",
                   help="Comma-separated metrics: faithfulness,answer_relevancy,"
                        "context_precision,context_recall  (default: all)")
    p.add_argument("--output-dir", type=str, default=str(Path(__file__).parent / "results"),
                   help="Directory to write result files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline_on_questions(questions: list[dict]) -> list[dict]:
    """
    Run the NPS pipeline on each question and collect answers + contexts.
    Returns a list of dicts with keys:
        question, answer, contexts, ground_truth, sanitized_query, latency_s
    """
    from pipeline.pipeline import NPSPipeline

    print("Loading NPS pipeline (takes ~1 min on first run)...")
    pipeline = NPSPipeline()
    print(f"Pipeline ready. Running {len(questions)} questions...\n")

    records = []
    for i, item in enumerate(questions, start=1):
        q = item["question"]
        print(f"[{i:02d}/{len(questions)}] {q[:70]}...")

        t0 = time.perf_counter()
        result = pipeline.query(q)
        latency = round(time.perf_counter() - t0, 2)

        # RAGAS expects contexts as a list of strings (one string per retrieved chunk)
        contexts = [chunk.text for chunk in result.reranked_chunks]

        records.append({
            "question":        q,
            "answer":          result.answer,
            "contexts":        contexts,
            "ground_truth":    item["ground_truth"],
            "sanitized_query": result.sanitized_query,
            "latency_s":       latency,
        })
        print(f"         latency: {latency}s | chunks: {len(contexts)}")

    return records


# ---------------------------------------------------------------------------
# RAGAS evaluation
# ---------------------------------------------------------------------------

def build_ragas_metrics(metric_names: list[str]):
    """Import and configure RAGAS metrics with the Groq LLM + BGE embeddings."""
    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Use small fast model for RAGAS — llama-3.1-8b-instant has 500k TPD
    # vs 100k TPD for 70b, so it won't hit daily limits during evaluation.
    llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.environ["GROQ_API_KEY"],
            temperature=0,
        )
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )

    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    all_metrics = {
        "faithfulness":      faithfulness,
        "answer_relevancy":  answer_relevancy,
        "context_precision": context_precision,
        "context_recall":    context_recall,
    }

    selected = {}
    for name in metric_names:
        name = name.strip()
        if name not in all_metrics:
            print(f"WARNING: unknown metric '{name}' — skipping")
            continue
        m = all_metrics[name]
        m.llm = llm
        if hasattr(m, "embeddings"):
            m.embeddings = emb
        selected[name] = m

    return list(selected.values())


def run_ragas(records: list[dict], metrics) -> dict:
    """Run RAGAS evaluate() and return the scores dict."""
    from datasets import Dataset
    from ragas import evaluate

    dataset = Dataset.from_dict({
        "question":     [r["question"]     for r in records],
        "answer":       [r["answer"]       for r in records],
        "contexts":     [r["contexts"]     for r in records],
        "ground_truth": [r["ground_truth"] for r in records],
    })

    from ragas.run_config import RunConfig

    print("\nRunning RAGAS evaluation (this calls the LLM for each sample)...")
    # max_workers=1 avoids Groq rate limits; timeout=120 handles slow responses
    result = evaluate(
        dataset,
        metrics=metrics,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1, timeout=120),
    )
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_results(
    records: list[dict],
    ragas_result,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── CSV with per-question scores ──────────────────────────────
    df = ragas_result.to_pandas()
    # Re-attach latency and sanitized_query from our records
    df["latency_s"]       = [r["latency_s"]       for r in records]
    df["sanitized_query"] = [r["sanitized_query"]  for r in records]

    csv_path = output_dir / f"ragas_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-question results saved → {csv_path}")

    # ── JSON with full context for debugging ──────────────────────
    json_path = output_dir / f"per_question_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Full question records saved → {json_path}")


def print_summary(ragas_result, records: list[dict]) -> None:
    scores = ragas_result.scores if hasattr(ragas_result, "scores") else {}
    aggregate = {k: float(v) for k, v in ragas_result.items() if isinstance(v, float)}

    latencies = [r["latency_s"] for r in records]
    avg_latency = sum(latencies) / len(latencies)

    col_w = 26
    print("\n" + "=" * 55)
    print(" RAGAS Evaluation Summary — NPS Chatbot")
    print("=" * 55)
    print(f"  Questions evaluated : {len(records)}")
    print(f"  Avg pipeline latency: {avg_latency:.2f}s")
    print("-" * 55)

    metric_descriptions = {
        "faithfulness":      "Answer grounded in context (0–1, higher=better)",
        "answer_relevancy":  "Answer addresses the question (0–1)",
        "context_precision": "Retrieved chunks are on-topic (0–1)",
        "context_recall":    "Context covers the ground truth (0–1)",
    }

    for metric, score in aggregate.items():
        if score != score:  # NaN check
            print(f"  {metric:<22} N/A    [rate limit or timeout — rerun]")
            continue
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {metric:<22} {score:.3f}  [{bar}]")

    print("=" * 55)

    # Flag low scores
    for metric, score in aggregate.items():
        if score == score and score < 0.5:  # skip NaN
            print(f"  ⚠  {metric} is below 0.5 — review retrieved chunks and system prompt.")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    from test_questions import TEST_CASES
    questions = TEST_CASES[:args.limit] if args.limit else TEST_CASES

    metric_names = (
        ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if args.metrics == "all"
        else args.metrics.split(",")
    )

    # Step 1 — run the NPS pipeline on all questions
    records = run_pipeline_on_questions(questions)

    # Step 2 — build RAGAS metrics (loads LLM + embeddings)
    metrics = build_ragas_metrics(metric_names)
    if not metrics:
        print("No valid metrics selected — aborting.")
        return

    # Step 3 — evaluate with RAGAS
    ragas_result = run_ragas(records, metrics)

    # Step 4 — display and save
    print_summary(ragas_result, records)
    save_results(records, ragas_result, Path(args.output_dir))


if __name__ == "__main__":
    main()
