"""
End-to-end smoke test for the knowledge-graph pipeline.

This script validates both RAG engines (LightRAG and RAGAnything, if installed)
on a small PDF subset in data/raw/papers/unparsed.

Run from any working directory:
    python examples/e2e_smoke_test.py
"""

import json
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project root (parent of examples/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from pgloop.knowledge import (  # noqa: E402
    RAGANYTHING_AVAILABLE,
    LightRAGEngine,
    RAGAnythingEngine,
)
from scripts.build_knowledge_graph import run_pipeline  # noqa: E402


def _clean_dir(path: Path):
    """Remove a directory if it exists."""
    if path.exists():
        print(f"Removing old directory: {path}")
        shutil.rmtree(path)


def _print_query_answers(engine_name: str, query_func):
    """Run two sample queries and print truncated answers."""
    questions = [
        "What are common phosphogypsum treatment pathways?",
        "How to extract rare earth elements from phosphogypsum?",
    ]
    print(f"\n{engine_name} query smoke test:")
    for q in questions:
        answer = query_func(q)
        print(f"\nQ: {q}\nA: {answer[:500]}")


def run_lightrag_smoke(limit: int):
    """Run LightRAG end-to-end smoke test."""
    lightrag_dir = PROJECT_ROOT / "data" / "processed" / "lightrag_db"
    _clean_dir(lightrag_dir)

    print("\n=== LightRAG smoke run ===")
    # Full pipeline for LightRAG: parse -> index -> extract -> build
    results = run_pipeline(
        steps=["all"],
        parser_type="pymupdf",
        limit=limit,
        engine="lightrag",
    )
    print("\nLightRAG pipeline results:")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    rag = LightRAGEngine(working_dir=lightrag_dir)
    _print_query_answers("LightRAG", lambda q: rag.query(q, mode="mix").answer)

    # Explicitly drop Python reference before moving to the next engine.
    # Note: Ollama model memory is managed by Ollama server, not by this script.
    del rag


def run_raganything_smoke(limit: int):
    """Run RAGAnything index/query smoke test if optional dependency is installed."""
    if not RAGANYTHING_AVAILABLE or RAGAnythingEngine is None:
        print("\n[SKIP] RAGAnything is not installed in this environment.")
        return

    raganything_dir = PROJECT_ROOT / "data" / "processed" / "raganything_db"
    _clean_dir(raganything_dir)

    print("\n=== RAGAnything smoke run ===")
    # RAGAnything parses PDFs internally during index step.
    results = run_pipeline(
        steps=["index"],
        parser_type="mineru",
        limit=limit,
        engine="raganything",
    )
    print("\nRAGAnything index results:")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    rag = RAGAnythingEngine(working_dir=raganything_dir)
    _print_query_answers("RAGAnything", lambda q: rag.query(q, mode="hybrid"))

    # Explicitly drop Python reference before script exit.
    # Note: Ollama model memory is managed by Ollama server, not by this script.
    del rag


def main():
    unparsed_dir = PROJECT_ROOT / "data" / "raw" / "papers" / "unparsed"
    pdfs = sorted(unparsed_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF found in: {unparsed_dir}")

    print(f"Found {len(pdfs)} PDFs in {unparsed_dir}")
    limit = min(2, len(pdfs))

    # Engines are executed sequentially in one process.
    run_lightrag_smoke(limit)
    run_raganything_smoke(limit)

    print("\nE2E smoke test completed.")


if __name__ == "__main__":
    main()
