import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pgloop.knowledge import LightRAGEngine


# Set dummy environment variables for testing
@pytest.fixture(autouse=True)
def setup_test_env():
    os.environ["LIGHTRAG_GRAPH_STORAGE"] = "Neo4JStorage"
    os.environ["LIGHTRAG_VECTOR_STORAGE"] = "MilvusVectorDBStorage"
    os.environ["MILVUS_URI"] = "http://localhost:19530"
    os.environ["MILVUS_DB_NAME"] = "test_db"
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "neo4j"
    os.environ["NEO4J_PASSWORD"] = "password"


@patch("pgloop.knowledge.lightrag_engine.LIGHTRAG_AVAILABLE", True)
@patch("pgloop.knowledge.lightrag_engine.LightRAG")
def test_lightrag_engine_init(mock_lightrag_cls):
    """Test that LightRAGEngine correctly forwards storage configuration to LightRAG."""
    engine = LightRAGEngine(
        working_dir="/tmp/test_lightrag",
        embedding_dim=2560,
    )

    # Trigger RAG instance creation
    with patch.object(engine, "_create_openai_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client_factory.return_value = mock_client
        _ = engine._get_rag_instance()

    # Assert correct parameters were passed to LightRAG
    mock_lightrag_cls.assert_called_once()
    kwargs = mock_lightrag_cls.call_args[1]

    assert kwargs["graph_storage"] == "Neo4JStorage"
    assert kwargs["vector_storage"] == "MilvusVectorDBStorage"
    assert kwargs["vector_db_storage_cls_kwargs"]["uri"] == "http://localhost:19530"
    assert kwargs["vector_db_storage_cls_kwargs"]["db_name"] == "test_db"
    assert kwargs["vector_db_storage_cls_kwargs"]["metric_type"] == "COSINE"


@patch("pgloop.knowledge.lightrag_engine.LIGHTRAG_AVAILABLE", True)
@patch("pgloop.knowledge.lightrag_engine.LightRAG")
def test_mrl_embedding_truncation(mock_lightrag_cls):
    """Test that embedding vectors are truncated correctly according to embedding_dim."""
    engine = LightRAGEngine(
        working_dir="/tmp/test_lightrag",
        embedding_dim=2560,
    )

    # Mock embedding API response
    mock_data = [MagicMock(embedding=[1.0] * 4096)]
    mock_response = MagicMock(data=mock_data)

    with patch.object(engine, "_create_openai_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_client_factory.return_value = mock_client

        embed_func = engine._create_embedding_func()

        # Test synchronous execution of async func
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            vectors = loop.run_until_complete(embed_func.func(["test text"]))
            assert vectors.shape == (1, 2560)
            assert np.all(vectors == 1.0)
        finally:
            loop.close()
