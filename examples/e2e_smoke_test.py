"""
End-to-end smoke test for the knowledge-graph pipeline.

This script validates both RAG engines (LightRAG and RAGAnything, if installed)
on a small PDF subset in data/raw/papers/unparsed.

LightRAG stage uses MinerU for PDF→Markdown by default (figures/tables; requires
`pip install -U "mineru[all]"` and model assets). Use `--lightrag-parser pymupdf`
for a faster text-only parse.

If `data/raw/papers/parsed/<stem>.md` already exists from an older PyMuPDF run,
`build_knowledge_graph` step 1 will skip parsing; delete that file to force MinerU.

Run from any working directory:
    python examples/e2e_smoke_test.py

Memory safety notes:
- Engines run sequentially, not in parallel.
- Python object cleanup alone does not unload Ollama model memory.
- To reduce GPU memory pressure, this script can call `ollama stop` between stages.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from typing import Any

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


def _lightrag_query_answer(engine: LightRAGEngine, q: str) -> str:
    """Single-query helper for smoke test (avoids ruff F821 on lambdas in try/finally)."""
    return engine.query(q, mode="mix").answer


def _raganything_query_answer(engine: Any, q: str) -> str:
    """Single-query helper for RAGAnything smoke test."""
    return engine.query(q, mode="hybrid")


def _clean_dir(path: Path):
    """Remove a directory if it exists."""
    if path.exists():
        print(f"Removing old directory: {path}")
        shutil.rmtree(path)


def _models_to_unload() -> list[str]:
    """Collect model names from env for best-effort unload."""
    models = []
    for model_name in [os.getenv("LLM_MODEL"), os.getenv("EMBEDDING_MODEL")]:
        if model_name and model_name not in models:
            models.append(model_name)
    return models


def _run_ollama_command(args: list[str]) -> subprocess.CompletedProcess | None:
    """Run an ollama CLI command, returning None if ollama is unavailable."""
    try:
        return subprocess.run(
            ["ollama", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return None


def _stop_ollama_models(models: list[str]):
    """
    Best-effort model unload to reduce VRAM pressure.

    NOTE:
    - `del` in Python only drops local references.
    - Actual model memory is controlled by the Ollama server process.
    """
    if not models:
        return
    for model in models:
        result = _run_ollama_command(["stop", model])
        if result is None:
            print("[WARN] `ollama` CLI not found. Cannot force-unload models.")
            return
        if result.returncode == 0:
            print(f"[INFO] Stopped Ollama model: {model}")
        else:
            # Non-zero is not fatal here (e.g., model already not running).
            err = (result.stderr or "").strip()
            if err:
                print(f"[WARN] Could not stop model `{model}`: {err}")


def _print_ollama_ps():
    """Print current loaded Ollama models for visibility."""
    result = _run_ollama_command(["ps"])
    if result is None:
        return
    output = (result.stdout or "").strip()
    if output:
        print("\nCurrent Ollama loaded models:")
        print(output)


def _memory_barrier(unload_ollama_models: bool):
    """
    Apply lightweight memory barrier between engine stages.

    1) Trigger Python garbage collection.
    2) Optionally ask Ollama to unload models to free VRAM.
    """
    gc.collect()
    if unload_ollama_models:
        _stop_ollama_models(_models_to_unload())
        _print_ollama_ps()


def _looks_like_oom(exc: Exception) -> bool:
    """Heuristic detection for out-of-memory errors."""
    msg = str(exc).lower()
    keywords = [
        "out of memory",
        "cuda out of memory",
        "oom",
        "resource exhausted",
        "insufficient memory",
    ]
    return any(k in msg for k in keywords)


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


def _has_index_success(results: dict) -> bool:
    """
    Check whether pipeline index step contains at least one successful document.

    Expected structure:
        {"index": {"file_a.md": True, "file_b.md": False, ...}, ...}
    """
    if not isinstance(results, dict):
        return False
    index_results = results.get("index")
    if not isinstance(index_results, dict) or not index_results:
        return False
    return any(bool(ok) for ok in index_results.values())


def run_lightrag_smoke(limit: int, parser_type: str = "mineru"):
    """Run LightRAG end-to-end smoke test (`parser_type`: mineru or pymupdf)."""
    lightrag_dir = PROJECT_ROOT / "data" / "processed" / "lightrag_db"
    _clean_dir(lightrag_dir)

    print("\n=== LightRAG smoke run ===")
    # Full pipeline for LightRAG: parse -> index -> extract -> build
    results = run_pipeline(
        steps=["all"],
        parser_type=parser_type,
        limit=limit,
        engine="lightrag",
    )
    print("\nLightRAG pipeline results:")
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))

    if not _has_index_success(results):
        print("\n[SKIP] LightRAG index failed. Skipping LightRAG query smoke.")
        return

    rag = LightRAGEngine(working_dir=lightrag_dir)
    try:
        _print_query_answers("LightRAG", partial(_lightrag_query_answer, rag))
    finally:
        if hasattr(rag, "close"):
            rag.close()
        # Drop Python reference before memory barrier.
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

    if not _has_index_success(results):
        print("\n[SKIP] RAGAnything index failed. Skipping RAGAnything query smoke.")
        return

    rag = RAGAnythingEngine(working_dir=raganything_dir)
    try:
        _print_query_answers("RAGAnything", partial(_raganything_query_answer, rag))
    finally:
        if hasattr(rag, "close"):
            rag.close()
        # Drop Python reference before memory barrier.
        del rag


def main():
    parser = argparse.ArgumentParser(description="Run LightRAG/RAGAnything smoke tests")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of PDFs to process per engine (default: 1 for lower VRAM risk)",
    )
    parser.add_argument(
        "--skip-raganything",
        action="store_true",
        help="Skip RAGAnything stage",
    )
    parser.add_argument(
        "--no-unload",
        action="store_true",
        help="Do not call `ollama stop` between stages",
    )
    parser.add_argument(
        "--lightrag-parser",
        choices=["mineru", "pymupdf"],
        default="mineru",
        help="PDF parser for LightRAG pipeline step 1 (default: mineru for images/tables)",
    )
    args = parser.parse_args()

    unparsed_dir = PROJECT_ROOT / "data" / "raw" / "papers" / "unparsed"
    pdfs = sorted(unparsed_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDF found in: {unparsed_dir}")

    print(f"Found {len(pdfs)} PDFs in {unparsed_dir}")
    limit = min(args.limit, len(pdfs))
    unload_between_stages = not args.no_unload

    # Engines are executed sequentially in one process.
    # Memory protection: clear Python refs + optional `ollama stop` between stages.
    run_lightrag_smoke(limit, parser_type=args.lightrag_parser)
    _memory_barrier(unload_between_stages)

    if not args.skip_raganything:
        try:
            run_raganything_smoke(limit)
        except Exception as exc:
            if _looks_like_oom(exc):
                print("\n[WARN] RAGAnything stage appears to hit OOM. Skipping it safely.")
                print(f"[WARN] Details: {exc}")
            else:
                raise
        finally:
            _memory_barrier(unload_between_stages)

    print("\nE2E smoke test completed.")


if __name__ == "__main__":
    main()
