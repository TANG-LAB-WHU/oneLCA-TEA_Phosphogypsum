"""
LightRAG Engine Module

Graph-enhanced Retrieval Augmented Generation engine for phosphogypsum literature.
Uses LightRAG for entity-relationship extraction and knowledge graph-based retrieval.

All LLM and embedding calls go through Ollama's OpenAI-compatible /v1 endpoint.

Configuration via environment variables (.env):
- LLM_BASE_URL: OpenAI-compatible API endpoint (default: http://127.0.0.1:11434/v1)
- LLM_API_KEY: API key (default: "ollama")
- LLM_MODEL: Chat model name (default: qwen3.5:35b)
- EMBEDDING_MODEL: Embedding model name (default: bge-m3:567m)
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

load_dotenv()

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc

    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    LightRAG = None
    QueryParam = None


@dataclass
class RetrievalResult:
    """Result from LightRAG retrieval."""

    query: str
    answer: str
    mode: str


@dataclass
class QueryResult:
    """Result from LightRAG query with context."""

    query: str
    answer: str
    mode: str
    sources: List[Dict] = None


class LightRAGEngine:
    """
    LightRAG-based Retrieval Augmented Generation engine.

    Features:
    - Entity-relationship extraction during indexing
    - Knowledge graph-based retrieval
    - Multiple query modes: local, global, hybrid, mix
    - Multimodal support: automatic image transcription (qwen3.5 vision)
    - All calls via Ollama's OpenAI-compatible /v1 endpoint (chat + embeddings)

    NOTE: EMBEDDING_MODEL must match the exact name shown by `ollama list`,
    e.g. "bge-m3:567m". The embedding dimension (default 1024) must also
    match the model's actual output dimension; adjust EMBEDDING_DIM if you
    switch to a different embedding model.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
    ):
        """
        Initialize the LightRAGEngine.

        Args:
            working_dir: Directory for storing graph and vector data
            llm_model: Chat model name (default: LLM_MODEL env or "qwen3.5:35b")
            embedding_model: Embedding model name (default: EMBEDDING_MODEL env or "bge-m3:567m")
            embedding_dim: Embedding vector dimension (default: EMBEDDING_DIM env or 1024)
            llm_base_url: OpenAI-compatible API base URL (default: LLM_BASE_URL env)
            llm_api_key: API key (default: LLM_API_KEY env or "ollama")
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError("lightrag not installed. Run: pip install lightrag-hku")

        self.working_dir = working_dir or Path("./data/processed/lightrag_db")
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # LLM configuration
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.llm_api_key = (
            llm_api_key
            or os.getenv("LLM_API_KEY")
            or "ollama"
        )
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen3.5:35b")

        # Embedding configuration — model name must match `ollama list`
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "1024"))

        self._rag = None

    def _get_rag_instance(self) -> LightRAG:
        """Create or return the LightRAG instance."""
        if self._rag is None:
            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=self._create_llm_func(),
                embedding_func=self._create_embedding_func(),
            )
        return self._rag

    async def _get_initialized_rag(self) -> LightRAG:
        """Get the rag instance and ensure it's initialized."""
        rag = self._get_rag_instance()
        await rag.initialize_storages()
        return rag

    def _create_llm_func(self):
        """Create LLM function via Ollama's /v1/chat/completions."""
        from openai import OpenAI

        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

        async def llm_func(
            prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(model=self.llm_model, messages=messages)
            return response.choices[0].message.content

        return llm_func

    def _create_embedding_func(self) -> Any:
        """Create embedding function via Ollama's /v1/embeddings."""
        import numpy as np
        from openai import OpenAI

        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

        async def embed_func(texts: list[str]) -> np.ndarray:
            response = client.embeddings.create(model=self.embedding_model, input=texts)
            return np.array([d.embedding for d in response.data])

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim, max_token_size=8192, func=embed_func
        )

    async def _transcribe_image(self, image_path: Path) -> str:
        """Transcribe an image using the LLM with vision capability."""
        try:
            import base64

            from openai import OpenAI

            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine image MIME type
            suffix = image_path.suffix.lower()
            mime_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                suffix.lstrip("."), "image/jpeg"
            )

            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This image is from a scientific paper about phosphogypsum. "
                                    "Describe it in detail (flowcharts, charts, tables). "
                                    "English only."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                            },
                        ],
                    }
                ],
            )
            return (
                f"\n\n[Image Analysis: {image_path.name}]\n{response.choices[0].message.content}\n"
            )
        except Exception as e:
            return f"\n\n[Error analyzing image {image_path.name}]: {str(e)}\n"

    def add_document(
        self, text: str, document_id: str = None, metadata: Dict[str, Any] = None
    ) -> None:
        """Add a document to the LightRAG database."""

        async def _work():
            rag = await self._get_initialized_rag()
            await rag.ainsert(text)

        asyncio.run(_work())

    def add_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.md",
        limit: int = None,
        transcribe_images: bool = True,
    ) -> Dict[str, bool]:
        """Add all documents from a directory, including image transcriptions."""
        directory = Path(directory)
        files = list(directory.rglob(pattern))
        if limit:
            files = files[:limit]

        results = {}

        async def _work():
            rag = await self._get_initialized_rag()
            for i, filepath in enumerate(files, 1):
                try:
                    print(f"[{i}/{len(files)}] Processing: {filepath.name[:50]}...")
                    with open(filepath, "r", encoding="utf-8") as f:
                        text = f.read()

                    if transcribe_images:
                        image_dirs = [filepath.parent / "images", filepath.parent.parent / "images"]
                        image_text = ""
                        for img_dir in image_dirs:
                            if img_dir.exists() and img_dir.is_dir():
                                img_files = list(img_dir.glob("*.[jJ][pP][gG]")) + list(
                                    img_dir.glob("*.[pP][nN][gG]")
                                )
                                for img_file in img_files[:5]:
                                    image_text += await self._transcribe_image(img_file)
                        if image_text:
                            text += "\n\n=== IMAGE DATA ===\n" + image_text

                    await rag.ainsert(text)
                    results[filepath.name] = True
                    print("    ✓ Indexed successfully")
                except Exception as e:
                    results[filepath.name] = False
                    print(f"    ✗ Error: {e}")

        asyncio.run(_work())
        return results

    def query(self, query: str, mode: str = "mix") -> QueryResult:
        """Query the knowledge base."""

        async def _work():
            rag = await self._get_initialized_rag()
            answer = await rag.aquery(query, param=QueryParam(mode=mode))
            return answer

        answer = asyncio.run(_work())
        return QueryResult(query=query, answer=answer, mode=mode)

    def retrieve(self, query: str, mode: str = "mix") -> RetrievalResult:
        """Retrieve relevant information (alias for query)."""
        result = self.query(query, mode)
        return RetrievalResult(query=result.query, answer=result.answer, mode=result.mode)

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            "working_dir": str(self.working_dir),
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "available": LIGHTRAG_AVAILABLE,
        }

    def delete_database(self) -> None:
        """Delete all data and reset the database."""
        import shutil

        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self._rag = None


def main():
    print("LightRAG Engine (Ollama-only)")
    print(f"  LightRAG available: {LIGHTRAG_AVAILABLE}")
    print("\nConfiguration from .env:")
    print(f"  LLM_BASE_URL:     {os.getenv('LLM_BASE_URL', '(not set)')}")
    print(f"  LLM_MODEL:        {os.getenv('LLM_MODEL', '(not set)')}")
    print(f"  EMBEDDING_MODEL:  {os.getenv('EMBEDDING_MODEL', '(not set)')}")
    print(f"  EMBEDDING_DIM:    {os.getenv('EMBEDDING_DIM', '1024')}")


if __name__ == "__main__":
    main()
