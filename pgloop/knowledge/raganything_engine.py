"""
RAGAnything Engine Module

All-in-one multimodal RAG engine for phosphogypsum literature processing.
Provides unified document parsing, multimodal understanding, and knowledge graph indexing.

Built on top of LightRAG with integrated MinerU parsing pipeline.
All LLM and embedding calls go through Ollama's OpenAI-compatible /v1 endpoint.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from dotenv import load_dotenv

try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    from raganything import RAGAnything, RAGAnythingConfig

    RAGANYTHING_AVAILABLE = True
except ImportError:
    RAGANYTHING_AVAILABLE = False

load_dotenv()


@dataclass
class MultimodalQueryResult:
    """Result from RAGAnything multimodal query."""

    query: str
    answer: str
    mode: str
    multimodal_context: Optional[List[Dict]] = None


class RAGAnythingEngine:
    """
    RAGAnything-based multimodal RAG engine.

    Provides enhanced multimodal processing compared to base LightRAG:
    - Integrated MinerU document parsing
    - Automatic table/formula/image extraction
    - Multimodal knowledge graph with cross-modal relations
    - Multimodal query support

    All calls via Ollama's OpenAI-compatible /v1 endpoint (chat + embeddings).

    NOTE: EMBEDDING_MODEL must match `ollama list` output (e.g. "bge-m3:567m").
    Adjust EMBEDDING_DIM if your model outputs a different vector size.
    """

    def __init__(
        self,
        working_dir: Optional[Path] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        parser: str = "mineru",
        parse_method: str = "auto",
    ):
        """
        Initialize RAGAnythingEngine.

        Args:
            working_dir: Directory for storing RAG data
            llm_model: Chat model name (default: LLM_MODEL env or "qwen3.5:35b")
            embedding_model: Embedding model name (default: EMBEDDING_MODEL env or "bge-m3:567m")
            embedding_dim: Embedding vector dimension (default: EMBEDDING_DIM env or 1024)
            llm_base_url: API base URL (default: LLM_BASE_URL env)
            llm_api_key: API key (default: LLM_API_KEY env or "ollama")
            parser: Document parser ("mineru" or "docling")
            parse_method: Parse method ("auto", "ocr", "txt")
        """
        if not RAGANYTHING_AVAILABLE:
            raise ImportError("RAGAnything not installed. Run: pip install 'raganything[all]'")

        self.working_dir = (
            Path(working_dir) if working_dir else Path("./data/processed/raganything_db")
        )
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

        # Parser configuration
        self.parser = parser
        self.parse_method = parse_method

        self._rag = None

    def _create_llm_func(self):
        """Create LLM function for RAGAnything."""

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                self.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                **kwargs,
            )

        return llm_model_func

    def _create_vision_func(self):
        """Create vision model function for multimodal processing."""
        llm_func = self._create_llm_func()

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            messages=None,
            **kwargs,
        ):
            # Multimodal VLM query with messages format
            if messages:
                return openai_complete_if_cache(
                    self.llm_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=messages,
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                    **kwargs,
                )
            # Single image format
            elif image_data:
                return openai_complete_if_cache(
                    self.llm_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        (
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}"
                                        },
                                    },
                                ],
                            }
                            if image_data
                            else {"role": "user", "content": prompt}
                        ),
                    ],
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                    **kwargs,
                )
            # Pure text
            else:
                return llm_func(prompt, system_prompt, history_messages, **kwargs)

        return vision_model_func

    def _create_embedding_func(self):
        """Create embedding function via Ollama's /v1/embeddings."""

        async def embed_func(texts: list[str]):
            embeddings = await openai_embed(
                texts,
                model=self.embedding_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
            return np.array(embeddings)

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim, max_token_size=8192, func=embed_func
        )

    def _get_rag_instance(self) -> Any:
        """Get or create RAGAnything instance."""
        if self._rag is None:
            config = RAGAnythingConfig(
                working_dir=str(self.working_dir),
                parser=self.parser,
                parse_method=self.parse_method,
                enable_image_processing=True,
                enable_table_processing=True,
                enable_equation_processing=True,
            )

            self._rag = RAGAnything(
                config=config,
                llm_model_func=self._create_llm_func(),
                vision_model_func=self._create_vision_func(),
                embedding_func=self._create_embedding_func(),
            )
        return self._rag

    async def aprocess_document(
        self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Asynchronously process a document with full multimodal extraction.
        """
        rag = self._get_rag_instance()
        file_path = Path(file_path)
        output_dir = Path(output_dir) if output_dir else self.working_dir / "parsed"

        return await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(output_dir),
            parse_method=self.parse_method,
            **kwargs,
        )

    def process_document(
        self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document with full multimodal extraction (Synchronous wrapper).
        """
        return asyncio.run(self.aprocess_document(file_path, output_dir, **kwargs))

    def process_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        limit: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, bool]:
        """
        Process all documents in a directory using a single event loop.
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        if limit:
            files = files[:limit]

        results = {}

        async def _process_all():
            for i, filepath in enumerate(files, 1):
                try:
                    print(f"[{i}/{len(files)}] Processing: {filepath.name[:50]}...")
                    await self.aprocess_document(filepath, **kwargs)
                    results[filepath.name] = True
                    print("    ✓ Processed successfully")
                except Exception as e:
                    results[filepath.name] = False
                    print(f"    ✗ Error: {e}")

        asyncio.run(_process_all())
        return results

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """Asynchronously query the knowledge base."""
        rag = self._get_rag_instance()
        return await rag.aquery(query, mode=mode)

    def query(self, query: str, mode: str = "hybrid") -> str:
        """Query the knowledge base (Synchronous wrapper)."""
        return asyncio.run(self.aquery(query, mode=mode))

    async def amultimodal_query(
        self, query: str, multimodal_content: Optional[List[Dict]] = None, mode: str = "hybrid"
    ) -> MultimodalQueryResult:
        """Asynchronously query with multimodal context."""
        rag = self._get_rag_instance()

        if multimodal_content:
            answer = await rag.aquery_with_multimodal(
                query, multimodal_content=multimodal_content, mode=mode
            )
        else:
            answer = await rag.aquery(query, mode=mode)

        return MultimodalQueryResult(
            query=query, answer=answer, mode=mode, multimodal_context=multimodal_content
        )

    def multimodal_query(
        self, query: str, multimodal_content: Optional[List[Dict]] = None, mode: str = "hybrid"
    ) -> MultimodalQueryResult:
        """Query with multimodal context (Synchronous wrapper)."""
        return asyncio.run(self.amultimodal_query(query, multimodal_content, mode))

    def check_parser_installation(self) -> bool:
        """Check if the parser (MinerU) is properly installed."""
        rag = self._get_rag_instance()
        return rag.check_parser_installation()

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "working_dir": str(self.working_dir),
            "parser": self.parser,
            "parse_method": self.parse_method,
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
        }


def main():
    print("RAGAnything Engine (Ollama-only)")
    print("-" * 40)
    print(f"RAGAnything available: {RAGANYTHING_AVAILABLE}")

    if RAGANYTHING_AVAILABLE:
        print("\nQuick test:")
        engine = RAGAnythingEngine()
        print(f"Parser installed: {engine.check_parser_installation()}")
        print(f"Stats: {engine.get_statistics()}")


if __name__ == "__main__":
    main()
