"""
Embeddings Module

Handles text embedding for RAG and similarity analysis.
"""

from typing import List, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingModel:
    """Wrapper for text embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.model = SentenceTransformer(model_name)
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text."""
        if not self.model:
            # Fallback to random if model not loaded (for dev)
            dim = 384
            count = 1 if isinstance(texts, str) else len(texts)
            return np.random.rand(count, dim).astype(np.float32)
        
        return self.model.encode(texts)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(emb1, emb2.T)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
