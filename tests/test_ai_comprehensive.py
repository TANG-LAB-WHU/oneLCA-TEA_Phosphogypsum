import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from dotenv import load_dotenv

from pgloop.knowledge.embeddings import EmbeddingModel
from pgloop.knowledge.gap_filler import GapFiller
from pgloop.knowledge.knowledge_graph import PhosphogypsumKG
from pgloop.knowledge.lightrag_engine import LightRAGEngine

# Import modules to test
from pgloop.knowledge.llm_extractor import LLMExtractor


@pytest.fixture(scope="module")
def setup_env():
    """Load environment variables and config."""
    load_dotenv()
    config_path = Path("config/settings.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture(scope="module")
def temp_test_dir():
    """Create a temporary directory for test data."""
    test_dir = Path("tests/temp_ai_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir
    if test_dir.exists():
        try:
            shutil.rmtree(test_dir)
        except PermissionError:
            print(
                f"\nWARNING: Could not remove {test_dir} due to PermissionError. "
                "This is common on Windows with ChromaDB."
            )


def test_embeddings():
    """Test embedding generation via Ollama and similarity."""
    if not os.getenv("LLM_BASE_URL"):
        pytest.skip("LLM_BASE_URL not set (optional integration test)")

    model = EmbeddingModel()
    text1 = "phosphogypsum treatment via cement production."
    text2 = "Using phosphogypsum as an additive in the cement industry."
    text3 = "Agricultural soil amendment using phosphogypsum."

    emb1 = model.encode(text1)
    emb2 = model.encode(text2)
    emb3 = model.encode(text3)

    expected_dim = int(os.getenv("EMBEDDING_DIM", "2560"))
    assert emb1.shape == (expected_dim,), f"Expected ({expected_dim},), got {emb1.shape}"

    sim12 = model.similarity(emb1, emb2)
    sim13 = model.similarity(emb1, emb3)

    print(f"\nSimilarity (Cement1 vs Cement2): {sim12:.4f}")
    print(f"Similarity (Cement1 vs Agriculture): {sim13:.4f}")

    assert sim12 > 0.4
    assert sim13 > 0.3


def test_llm_extraction(setup_env):
    """Test LLM-based data extraction (OpenAI-compatible API)."""
    if not os.getenv("LLM_BASE_URL"):
        pytest.skip("LLM_BASE_URL not found in .env (optional integration test)")

    extractor = LLMExtractor()

    text = (
        "Detailed analysis of PG from Florida shows 94% CaSO4, 0.5% P2O5, and Ra-226 at 200 Bq/kg."
    )
    result = extractor.extract(text, "composition")

    assert result.success is True
    assert result.data["CaSO4"] is not None
    print(f"\nExtracted JSON: {result.data}")


def test_rag_flow(setup_env, temp_test_dir):
    """Test LightRAG indexing and search."""
    if not os.getenv("LLM_BASE_URL"):
        pytest.skip("LLM_BASE_URL not found in .env (optional integration test)")

    rag = LightRAGEngine(working_dir=temp_test_dir / "lightrag")

    # Index document
    doc_text = (
        "The REE extraction process from phosphogypsum involves sulfuric acid "
        "leaching followed by solvent extraction using D2EHPA."
    )
    rag.add_document(doc_text)

    # Query (LightRAG query already includes generation and relationship reasoning)
    query = "How to extract rare earth elements from PG?"
    result = rag.query(query, mode="local")

    assert result.answer is not None
    print(f"\nLightRAG Answer: {result.answer}")


def test_knowledge_graph(temp_test_dir):
    """Test Knowledge Graph operations."""
    kg = PhosphogypsumKG(storage_path=temp_test_dir / "kg")

    # Add nodes
    kg.add_country("Brazil", "South America")
    kg.add_composition("Brazil_Sample", "Brazil", CaSO4=0.90, P2O5=0.02)

    # Check structure
    stats = kg.get_statistics()
    assert stats["total_nodes"] >= 2
    assert "produces" in stats["edges_by_relation"]

    # Check data gaps
    gaps = kg.find_data_gaps()
    assert "Composition" in gaps
    print(f"\nKG Gaps: {gaps}")

    kg.save_graph()
    assert (temp_test_dir / "kg" / "kg_nodes.json").exists()


def test_gap_filler():
    """Test ML-based gap filling."""
    filler = GapFiller()

    # Training-like data
    filler.add_reference_data("Case A", {"capex": 100, "opex": 10, "capacity": 1000})
    filler.add_reference_data("Case B", {"capex": 200, "opex": 18, "capacity": 2000})
    filler.add_reference_data("Case C", {"capex": 50, "opex": 6, "capacity": 500})

    # Predict
    known = {"capacity": 1500}
    result = filler.predict_by_similarity(known, "capex")

    assert not np.isnan(result.predicted_value)
    assert 100 <= result.predicted_value <= 200
    print(f"\nGap Filler Prediction (Capex for 1500t): {result.predicted_value:.2f}")


def main():
    # If run directly, execute all tests
    print("Starting Comprehensive AI Module Test...")
    pytest.main([__file__, "-s", "-v"])


if __name__ == "__main__":
    main()
