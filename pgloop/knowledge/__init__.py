"""
Provides knowledge graph construction, LLM extraction, and RAG capabilities.
"""

from pgloop.knowledge.embeddings import EmbeddingModel
from pgloop.knowledge.gap_filler import GapFiller
from pgloop.knowledge.graph import PhosphogypsumKG, Neo4jAdapter, Neo4jConfig, NEO4J_AVAILABLE
from pgloop.knowledge.llm_extractor import LLMExtractor

try:
    from pgloop.knowledge.lightrag_engine import LIGHTRAG_AVAILABLE, LightRAGEngine
except ImportError:
    LightRAGEngine = None
    LIGHTRAG_AVAILABLE = False

try:
    from pgloop.knowledge.raganything_engine import RAGANYTHING_AVAILABLE, RAGAnythingEngine
except ImportError:
    RAGAnythingEngine = None
    RAGANYTHING_AVAILABLE = False

__all__ = [
    "PhosphogypsumKG",
    "LLMExtractor",
    "LightRAGEngine",
    "RAGAnythingEngine",
    "GapFiller",
    "EmbeddingModel",
    "Neo4jAdapter",
    "Neo4jConfig",
    "NEO4J_AVAILABLE",
    "LIGHTRAG_AVAILABLE",
    "RAGANYTHING_AVAILABLE",
]
