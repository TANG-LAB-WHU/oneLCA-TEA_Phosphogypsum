"""
LightRAG Engine Module

Graph-enhanced Retrieval Augmented Generation engine for phosphogypsum literature.
Uses LightRAG for entity-relationship extraction and knowledge graph-based retrieval.

All LLM and embedding calls go through Ollama's OpenAI-compatible /v1 endpoint.

Configuration via environment variables (.env):
- LLM_BASE_URL: OpenAI-compatible API endpoint (default: http://127.0.0.1:11434/v1)
- LLM_API_KEY: API key (default: "ollama")
- LLM_MODEL: Chat model name (default: qwen3.5:35b)
- LLM_TEMPERATURE: Sampling temperature for chat (default: 0.1). LightRAG entity
  extraction does not pass temperature; without this, the OpenAI client defaults
  to ~1.0 and often breaks strict tuple-delimited output (warnings like 5/4 fields).
- EMBEDDING_MODEL: Embedding model name (default: qwen3-embedding:4b)
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

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


def _read_env_int(*names: str, default: int = 0) -> int:
    """Read the first valid integer env var from names."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _read_env_float(*names: str, default: float = 0.0) -> float:
    """Read the first valid float env var from names."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return default


def _read_env_bool(*names: str, default: bool = False) -> bool:
    """Read the first valid boolean env var from names."""
    for name in names:
        raw = os.getenv(name)
        if raw is None or str(raw).strip() == "":
            continue
        return str(raw).strip().lower() not in {"0", "false", "off", "no"}
    return default


def _is_local_base_url(url: str) -> bool:
    """Return True when the API host points to localhost."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host in {"127.0.0.1", "localhost", "::1"}


def _normalize_timeout(value: float) -> Optional[float]:
    """Treat non-positive timeout as no timeout override."""
    return value if value > 0 else None


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
    e.g. "qwen3-embedding:4b". The embedding dimension (default 2560) must also
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
            embedding_model: Embedding model name
                (default: EMBEDDING_MODEL env or "qwen3-embedding:4b")
            embedding_dim: Embedding vector dimension (default: EMBEDDING_DIM env or 2560)
            llm_base_url: OpenAI-compatible API base URL (default: LLM_BASE_URL env)
            llm_api_key: API key (default: LLM_API_KEY env or "ollama")
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError("lightrag not installed. Run: pip install lightrag-hku")

        self.working_dir = working_dir or Path("./data/processed/lightrag_db")
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # LLM configuration
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY") or "ollama"
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "qwen3.5:35b")
        self.llm_timeout = _normalize_timeout(_read_env_float("LLM_TIMEOUT", default=180.0))

        # Embedding configuration — model name must match `ollama list`
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "qwen3-embedding:4b")
        self.embedding_dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "2560"))
        self.embedding_timeout = _normalize_timeout(
            _read_env_float("EMBEDDING_TIMEOUT", default=30.0)
        )
        # Request-level context length override for Ollama (/v1 compatible path).
        # Prefer explicit LLM_CONTEXT_LENGTH and fallback to OLLAMA_CONTEXT_LENGTH.
        self.llm_context_length = _read_env_int(
            "LLM_CONTEXT_LENGTH", "OLLAMA_CONTEXT_LENGTH", default=0
        )
        # LightRAG's extract path calls llm_model_func without temperature; OpenAI's
        # default (~1.0) yields noisy delimiter-based records. Use a low default.
        raw_temp = os.getenv("LLM_TEMPERATURE", "0.1").strip()
        try:
            self.llm_temperature = float(raw_temp)
        except ValueError:
            self.llm_temperature = 0.1
        # For localhost Ollama, default to not inheriting system HTTP proxy env vars.
        default_trust_env = not _is_local_base_url(self.llm_base_url)
        self.llm_trust_env = _read_env_bool("LLM_TRUST_ENV", default=default_trust_env)

        self._rag = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_or_create_loop(self) -> asyncio.AbstractEventLoop:
        """
        Return a dedicated loop for this engine instance.

        Reusing one loop avoids cross-loop object binding issues in LightRAG workers.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run_coroutine(self, coro):
        """Run coroutine on the engine's dedicated event loop."""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            raise RuntimeError(
                "LightRAGEngine sync API called from an active event loop. "
                "Use async methods in this context."
            )

        loop = self._get_or_create_loop()
        return loop.run_until_complete(coro)

    def _build_extra_body(
        self, existing: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Merge request extra_body with Ollama context hints when configured."""
        extra_body: Dict[str, Any] = dict(existing or {})
        if self.llm_context_length > 0:
            options = dict(extra_body.get("options") or {})
            options.setdefault("num_ctx", self.llm_context_length)
            extra_body["options"] = options
        return extra_body or None

    @staticmethod
    def _sanitize_embedding_text(text: str) -> str:
        """Normalize problematic characters before embedding retry."""
        cleaned = (text or "").replace("\x00", " ").replace("\ufeff", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned else " "

    def close(self):
        """Close dedicated resources held by this engine instance."""
        if self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._loop = None

    def _create_openai_client(self, timeout: Optional[float]):
        """Create OpenAI-compatible client with aligned timeout/proxy behavior."""
        from openai import DefaultHttpxClient, OpenAI

        client_kwargs: Dict[str, Any] = {
            "base_url": self.llm_base_url,
            "api_key": self.llm_api_key,
        }
        http_client_kwargs: Dict[str, Any] = {"trust_env": self.llm_trust_env}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
            http_client_kwargs["timeout"] = timeout

        try:
            client_kwargs["http_client"] = DefaultHttpxClient(**http_client_kwargs)
        except TypeError:
            # Fallback for older SDK variants with narrower client kwargs support.
            pass

        return OpenAI(**client_kwargs)

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
        client = self._create_openai_client(timeout=self.llm_timeout)
        passthrough_keys = {
            "temperature",
            "max_tokens",
            "top_p",
            "stop",
            "seed",
            "response_format",
            "timeout",
            "stream",
            "tools",
            "tool_choice",
        }

        async def llm_func(
            prompt: str, system_prompt: str = None, history_messages: list = None, **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            request_kwargs: Dict[str, Any] = {"model": self.llm_model, "messages": messages}
            for key in passthrough_keys:
                if key in kwargs and kwargs[key] is not None:
                    request_kwargs[key] = kwargs[key]
            if "temperature" not in request_kwargs:
                request_kwargs["temperature"] = self.llm_temperature
            extra_body = self._build_extra_body(kwargs.get("extra_body"))
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)
            return response.choices[0].message.content

        return llm_func

    def _create_embedding_func(self) -> Any:
        """Create embedding function via Ollama's /v1/embeddings."""
        import numpy as np

        client = self._create_openai_client(timeout=self.embedding_timeout)

        def _embed_batch(batch_texts: list[str]) -> np.ndarray:
            response = client.embeddings.create(model=self.embedding_model, input=batch_texts)
            vectors = np.array([d.embedding for d in response.data], dtype=np.float32)
            if vectors.shape[0] != len(batch_texts):
                n = len(batch_texts)
                got = vectors.shape[0]
                raise ValueError(f"Embedding response size mismatch: expected {n}, got {got}")
            if not np.isfinite(vectors).all():
                raise ValueError("Embedding response contains non-finite values (NaN/Inf).")
            return vectors

        async def embed_func(texts: list[str]) -> np.ndarray:
            try:
                return _embed_batch(texts)
            except Exception as batch_error:
                # Retry once with normalized text, then isolate problematic inputs.
                cleaned_texts = [self._sanitize_embedding_text(t) for t in texts]
                if cleaned_texts != texts:
                    try:
                        return _embed_batch(cleaned_texts)
                    except Exception:
                        pass

                vectors: list[np.ndarray] = []
                failures: list[str] = []
                for idx, text in enumerate(cleaned_texts):
                    try:
                        vectors.append(_embed_batch([text])[0])
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

    async def _transcribe_image(self, image_path: Path) -> str:
        """Transcribe an image using the LLM with vision capability."""
        try:
            import base64

            client = self._create_openai_client(timeout=self.llm_timeout)

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Determine image MIME type
            suffix = image_path.suffix.lower()
            mime_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
                suffix.lstrip("."), "image/jpeg"
            )

            request_kwargs: Dict[str, Any] = {
                "model": self.llm_model,
                "messages": [
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
                "temperature": self.llm_temperature,
            }
            extra_body = self._build_extra_body()
            if extra_body:
                request_kwargs["extra_body"] = extra_body

            response = client.chat.completions.create(**request_kwargs)
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

        self._run_coroutine(_work())

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
                        image_dirs = [
                            filepath.parent / "images",
                            filepath.parent.parent / "images",
                        ]
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

        self._run_coroutine(_work())
        return results

    def query(self, query: str, mode: str = "mix") -> QueryResult:
        """Query the knowledge base."""

        async def _work():
            rag = await self._get_initialized_rag()
            answer = await rag.aquery(query, param=QueryParam(mode=mode))
            return answer

        answer = self._run_coroutine(_work())
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
            "llm_context_length": self.llm_context_length,
            "llm_temperature": self.llm_temperature,
            "llm_timeout": self.llm_timeout,
            "embedding_timeout": self.embedding_timeout,
            "llm_trust_env": self.llm_trust_env,
            "available": LIGHTRAG_AVAILABLE,
        }

    def delete_database(self) -> None:
        """Delete all data and reset the database."""
        import shutil

        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self._rag = None
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Avoid destructor-time exceptions.
            pass


def main():
    print("LightRAG Engine (Ollama-only)")
    print(f"  LightRAG available: {LIGHTRAG_AVAILABLE}")
    print("\nConfiguration from .env:")
    print(f"  LLM_BASE_URL:     {os.getenv('LLM_BASE_URL', '(not set)')}")
    print(f"  LLM_MODEL:        {os.getenv('LLM_MODEL', '(not set)')}")
    print(f"  EMBEDDING_MODEL:  {os.getenv('EMBEDDING_MODEL', '(not set)')}")
    print(f"  EMBEDDING_DIM:    {os.getenv('EMBEDDING_DIM', '2560')}")


if __name__ == "__main__":
    main()
