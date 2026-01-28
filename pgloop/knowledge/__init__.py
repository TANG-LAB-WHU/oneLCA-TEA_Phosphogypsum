"""
Knowledge Graph and AI Module

Provides knowledge graph construction, LLM extraction, and RAG capabilities.
"""

from pgloop.knowledge.knowledge_graph import PhosphogypsumKG
from pgloop.knowledge.llm_extractor import LLMExtractor
from pgloop.knowledge.gap_filler import GapFiller
from pgloop.knowledge.embeddings import EmbeddingModel

# LightRAG-based RAG engine (replaces ChromaDB)
try:
    from pgloop.knowledge.lightrag_engine import (
        LightRAGEngine, 
        LIGHTRAG_AVAILABLE
    )
except ImportError:
    LightRAGEngine = None
    LIGHTRAG_AVAILABLE = False

# Optional Neo4j support
try:
    from pgloop.knowledge.neo4j_adapter import Neo4jAdapter, Neo4jConfig
    NEO4J_AVAILABLE = True
except ImportError:
    Neo4jAdapter = None
    Neo4jConfig = None
    NEO4J_AVAILABLE = False

__all__ = [
    "PhosphogypsumKG",
    "LLMExtractor",
    "LightRAGEngine",
    "GapFiller",
    "EmbeddingModel",
    "Neo4jAdapter",
    "Neo4jConfig",
    "NEO4J_AVAILABLE",
    "LIGHTRAG_AVAILABLE",
]
