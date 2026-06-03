"""
Handles text embedding for RAG and similarity analysis.

Calls Ollama's OpenAI-compatible /v1/embeddings endpoint.
Configure EMBEDDING_MODEL (must match `ollama list`) and EMBEDDING_DIM in .env.
"""

import os
from typing import List, Union

import numpy as np
from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel:
    """Wrapper around Ollama's /v1/embeddings endpoint."""

    def __init__(
        self,
        model_name: str = None,
        base_url: str = None,
        api_key: str = None,
    ):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.dim = int(os.getenv("EMBEDDING_DIM", "2560"))
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings via Ollama /v1/embeddings."""
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        client = self._get_client()
        response = client.embeddings.create(model=self.model_name, input=texts)
        embeddings = np.array([d.embedding for d in response.data], dtype=np.float32)

        if single:
            return embeddings[0]
        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity."""
        dot = np.dot(emb1, emb2.T)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return float(dot / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
