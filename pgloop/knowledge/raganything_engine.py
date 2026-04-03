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


def _read_env_int(*names: str, default: int = 0) -> int:
    """Read first valid integer environment variable."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


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

    LLM_TEMPERATURE (default 0.1) is applied when callers omit temperature, matching
    LightRAGEngine so delimiter-structured LightRAG/RAGAnything extractions stay stable.

    NOTE: EMBEDDING_MODEL must match `ollama list` output (e.g. "qwen3-embedding:4b").
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
            embedding_model: Embedding model name
                (default: EMBEDDING_MODEL env or "qwen3-embedding:4b")
            embedding_dim: Embedding vector dimension (default: EMBEDDING_DIM env or 2560)
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
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY") or "ollama"
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen3.5:35b")

        # Embedding configuration — model name must match `ollama list`
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "2560"))
        self.llm_context_length = _read_env_int(
            "LLM_CONTEXT_LENGTH", "OLLAMA_CONTEXT_LENGTH", default=0
        )
        raw_temp = os.getenv("LLM_TEMPERATURE", "0.1").strip()
        try:
            self.llm_temperature = float(raw_temp)
        except ValueError:
            self.llm_temperature = 0.1

        # Parser configuration
        self.parser = parser
        self.parse_method = parse_method
        # Optional parser knobs for MinerU. Only applied when explicitly configured.
        self.mineru_backend = os.getenv("RAGANYTHING_MINERU_BACKEND")
        self.mineru_device = os.getenv("RAGANYTHING_MINERU_DEVICE")

        self._rag = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """Return a dedicated loop for this engine instance."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_coroutine(self, coro):
        """Run coroutine with a stable, instance-scoped event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            raise RuntimeError(
                "RAGAnythingEngine sync API called from an active event loop. "
                "Use async methods in this context."
            )

        loop = self._get_or_create_loop()
        return loop.run_until_complete(coro)

    def _build_extra_body(
        self, existing: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Attach Ollama context hints without discarding caller-provided extra_body."""
        extra_body: Dict[str, Any] = dict(existing or {})
        if self.llm_context_length > 0:
            options = dict(extra_body.get("options") or {})
            options.setdefault("num_ctx", self.llm_context_length)
            extra_body["options"] = options
        return extra_body or None

    @staticmethod
    def _sanitize_embedding_text(text: str) -> str:
        """Normalize text for retry when embedding request fails."""
        cleaned = (text or "").replace("\x00", " ").replace("\ufeff", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned else " "

    def close(self):
        """Close resources owned by this engine instance."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    def _create_llm_func(self):
        """Create LLM function for RAGAnything."""

        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            request_kwargs = dict(kwargs)
            if request_kwargs.get("temperature") is None:
                request_kwargs["temperature"] = self.llm_temperature
            extra_body = self._build_extra_body(request_kwargs.get("extra_body"))
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            return openai_complete_if_cache(
                self.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
                **request_kwargs,
            )

        return llm_model_func

    def _create_vision_func(self):
        """Create vision model function for multimodal processing."""
        llm_func = self._create_llm_func()

        def vision_model_func(
            prompt,
            system_prompt=None,
            history_messages=None,
            image_data=None,
            messages=None,
            **kwargs,
        ):
            request_kwargs = dict(kwargs)
            if request_kwargs.get("temperature") is None:
                request_kwargs["temperature"] = self.llm_temperature
            extra_body = self._build_extra_body(request_kwargs.get("extra_body"))
            if extra_body:
                request_kwargs["extra_body"] = extra_body

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
                    **request_kwargs,
                )
            # Single image format
            elif image_data:
                mm_messages = []
                if system_prompt:
                    mm_messages.append({"role": "system", "content": system_prompt})
                mm_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                            },
                        ],
                    }
                )
                return openai_complete_if_cache(
                    self.llm_model,
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=mm_messages,
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                    **request_kwargs,
                )
            # Pure text
            else:
                return llm_func(prompt, system_prompt, history_messages or [], **request_kwargs)

        return vision_model_func

    def _create_embedding_func(self):
        """Create embedding function via Ollama's /v1/embeddings."""

        async def _embed_batch(batch_texts: list[str]) -> np.ndarray:
            embeddings = await openai_embed(
                batch_texts,
                model=self.embedding_model,
                api_key=self.llm_api_key,
                base_url=self.llm_base_url,
            )
            vectors = np.array(embeddings, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = np.expand_dims(vectors, axis=0)
            if vectors.shape[0] != len(batch_texts):
                n = len(batch_texts)
                got = vectors.shape[0]
                raise ValueError(f"Embedding response size mismatch: expected {n}, got {got}")
            if not np.isfinite(vectors).all():
                raise ValueError("Embedding response contains non-finite values (NaN/Inf).")
            return vectors

        async def embed_func(texts: list[str]):
            try:
                return await _embed_batch(texts)
            except Exception as batch_error:
                cleaned_texts = [self._sanitize_embedding_text(t) for t in texts]
                if cleaned_texts != texts:
                    try:
                        return await _embed_batch(cleaned_texts)
                    except Exception:
                        pass

                vectors: list[np.ndarray] = []
                failures: list[str] = []
                for idx, text in enumerate(cleaned_texts):
                    try:
                        vectors.append((await _embed_batch([text]))[0])
                    except Exception as single_error:
                        failures.append(f"idx={idx}: {single_error}")

                if failures:
                    raise RuntimeError(
                        "Embedding failed after batch retry. "
                        f"Failed inputs: {len(failures)}/{len(cleaned_texts)}; "
                        + " | ".join(failures[:5])
                    ) from batch_error

                return np.vstack(vectors).astype(np.float32)

        return EmbeddingFunc(
            embedding_dim=self.embedding_dim,
            max_token_size=8192,
            func=embed_func,
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
        process_kwargs = dict(kwargs)
        if self.mineru_backend and "backend" not in process_kwargs:
            process_kwargs["backend"] = self.mineru_backend
        if self.mineru_device and "device" not in process_kwargs:
            process_kwargs["device"] = self.mineru_device

        return await rag.process_document_complete(
            file_path=str(file_path),
            output_dir=str(output_dir),
            parse_method=self.parse_method,
            **process_kwargs,
        )

    def process_document(
        self, file_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document with full multimodal extraction (Synchronous wrapper).
        """
        return self._run_coroutine(self.aprocess_document(file_path, output_dir, **kwargs))

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

        self._run_coroutine(_process_all())
        return results

    async def aquery(self, query: str, mode: str = "hybrid") -> str:
        """Asynchronously query the knowledge base."""
        rag = self._get_rag_instance()
        return await rag.aquery(query, mode=mode)

    def query(self, query: str, mode: str = "hybrid") -> str:
        """Query the knowledge base (Synchronous wrapper)."""
        return self._run_coroutine(self.aquery(query, mode=mode))

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
        return self._run_coroutine(self.amultimodal_query(query, multimodal_content, mode))

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
            "llm_temperature": self.llm_temperature,
            "llm_context_length": self.llm_context_length,
            "mineru_backend": self.mineru_backend,
            "mineru_device": self.mineru_device,
        }

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid destructor-time exceptions.
            pass


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
