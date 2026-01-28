"""
LightRAG Engine Module

Graph-enhanced Retrieval Augmented Generation engine for phosphogypsum literature.
Uses LightRAG for entity-relationship extraction and knowledge graph-based retrieval.

Configuration via environment variables (.env):
- LLM_BASE_URL: OpenAI-compatible API endpoint (default: proxy for Gemini)
- LLM_API_KEY: API key for the LLM service
- LLM_MODEL: Model name (e.g., gemini-3-flash, gpt-4o)
- EMBEDDING_MODEL: bge-m3 (local, free) or text-embedding-004 (API)
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from dotenv import load_dotenv
import PIL.Image

# Load environment variables
load_dotenv()

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    LightRAG = None
    QueryParam = None

# Check for local embedding model availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


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
    - Multimodal support: automatic image transcription
    - Unified OpenAI-compatible LLM interface (supports Gemini via proxy)
    - Dual embedding: local bge-m3 (free) or API-based
    """
    
    # Supported embedding models
    EMBEDDING_MODELS = {
        "bge-m3": {"dim": 1024, "type": "local", "model_name": "BAAI/bge-m3"},
        "text-embedding-004": {"dim": 768, "type": "gemini", "model_name": "models/text-embedding-004"},
        "text-embedding-3-small": {"dim": 1536, "type": "openai", "model_name": "text-embedding-3-small"},
        "text-embedding-3-large": {"dim": 3072, "type": "openai", "model_name": "text-embedding-3-large"},
    }
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None
    ):
        """
        Initialize the LightRAGEngine.
        
        Args:
            working_dir: Directory for storing graph and vector data
            llm_model: LLM model name (default: from LLM_MODEL env var)
            embedding_model: Embedding model (default: from EMBEDDING_MODEL env var or "bge-m3")
            llm_base_url: OpenAI-compatible API base URL (default: from LLM_BASE_URL env var)
            llm_api_key: API key for LLM service (default: from LLM_API_KEY env var)
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError("lightrag not installed. Run: pip install lightrag-hku")
        
        self.working_dir = working_dir or Path("./data/processed/lightrag_db")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM configuration (environment-driven with overrides)
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8045/v1")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gemini-3-flash")
        
        # Embedding configuration
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        
        # Validate embedding model
        if self.embedding_model not in self.EMBEDDING_MODELS:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}. Choose from: {list(self.EMBEDDING_MODELS.keys())}")
        
        model_info = self.EMBEDDING_MODELS[self.embedding_model]
        if model_info["type"] == "local" and not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        # For API-based embeddings (Gemini)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Local embedding model instance (lazy loaded)
        self._local_embed_model = None
        
        # LightRAG instance
        self._rag = None
    
    def _get_local_embed_model(self) -> "SentenceTransformer":
        """Lazy load the local embedding model."""
        if self._local_embed_model is None:
            model_name = self.EMBEDDING_MODELS[self.embedding_model]["model_name"]
            print(f"Loading local embedding model: {model_name}")
            self._local_embed_model = SentenceTransformer(model_name)
            device = self._local_embed_model.device.type if hasattr(self._local_embed_model.device, 'type') else str(self._local_embed_model.device)
            print(f"Model loaded on: {device.upper()}")
        return self._local_embed_model
    
    def _get_rag_instance(self) -> LightRAG:
        """Create or return the LightRAG instance."""
        if self._rag is None:
            # LLM function (OpenAI-compatible)
            llm_func = self._create_openai_compatible_llm_func()
            
            # Embedding function
            model_info = self.EMBEDDING_MODELS[self.embedding_model]
            if model_info["type"] == "local":
                embed_func = self._create_local_embed_func()
            elif model_info["type"] == "gemini":
                embed_func = self._create_gemini_embed_func()
            else:
                embed_func = self._create_openai_embed_func()
            
            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_func,
                embedding_func=embed_func
            )
        return self._rag

    async def _get_initialized_rag(self) -> LightRAG:
        """Get the rag instance and ensure it's initialized."""
        rag = self._get_rag_instance()
        await rag.initialize_storages()
        return rag

    def _create_openai_compatible_llm_func(self):
        """Create LLM function using OpenAI-compatible interface."""
        from openai import OpenAI
        
        client = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key
        )
        
        async def llm_func(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            **kwargs
        ) -> str:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=messages
            )
            return response.choices[0].message.content
        
        return llm_func

    def _create_local_embed_func(self) -> EmbeddingFunc:
        """Create embedding function for local bge-m3 model."""
        import numpy as np
        model_info = self.EMBEDDING_MODELS[self.embedding_model]
        
        async def embed_func(texts: list[str]) -> np.ndarray:
            model = self._get_local_embed_model()
            embeddings = model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings)
        
        return EmbeddingFunc(
            embedding_dim=model_info["dim"],
            max_token_size=8192,
            func=embed_func
        )

    def _create_gemini_embed_func(self) -> EmbeddingFunc:
        """Create embedding function for Gemini API."""
        import google.generativeai as genai
        import numpy as np
        
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY required for text-embedding-004")
        
        genai.configure(api_key=self.gemini_api_key)
        model_info = self.EMBEDDING_MODELS[self.embedding_model]
        
        async def embed_func(texts: list[str]) -> np.ndarray:
            embeddings = []
            for text in texts:
                result = genai.embed_content(model=model_info["model_name"], content=text)
                embeddings.append(result['embedding'])
            return np.array(embeddings)
        
        return EmbeddingFunc(embedding_dim=model_info["dim"], max_token_size=8192, func=embed_func)

    def _create_openai_embed_func(self) -> EmbeddingFunc:
        """Create embedding function for OpenAI API."""
        from openai import OpenAI
        import numpy as np
        
        client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
        model_info = self.EMBEDDING_MODELS[self.embedding_model]
        
        async def embed_func(texts: list[str]) -> np.ndarray:
            response = client.embeddings.create(
                model=model_info["model_name"],
                input=texts
            )
            return np.array([d.embedding for d in response.data])
        
        return EmbeddingFunc(embedding_dim=model_info["dim"], max_token_size=8192, func=embed_func)

    async def _transcribe_image(self, image_path: Path) -> str:
        """Transcribe an image using the LLM with vision capability."""
        try:
            from openai import OpenAI
            import base64
            
            client = OpenAI(base_url=self.llm_base_url, api_key=self.llm_api_key)
            
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
            
            # Determine image MIME type
            suffix = image_path.suffix.lower()
            mime_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(suffix.lstrip("."), "image/jpeg")
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "This image is from a scientific paper about phosphogypsum. Describe it in detail (flowcharts, charts, tables). English only."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}}
                    ]
                }]
            )
            return f"\n\n[Image Analysis: {image_path.name}]\n{response.choices[0].message.content}\n"
        except Exception as e:
            return f"\n\n[Error analyzing image {image_path.name}]: {str(e)}\n"

    def add_document(self, text: str, document_id: str = None, metadata: Dict[str, Any] = None) -> None:
        """Add a document to the LightRAG database."""
        async def _work():
            rag = await self._get_initialized_rag()
            await rag.ainsert(text)
        asyncio.run(_work())
    
    def add_documents_from_directory(
        self, directory: Union[str, Path], pattern: str = "*.md", limit: int = None, transcribe_images: bool = True
    ) -> Dict[str, bool]:
        """Add all documents from a directory, including image transcriptions."""
        directory = Path(directory)
        files = list(directory.rglob(pattern))
        if limit: files = files[:limit]
        
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
                                img_files = list(img_dir.glob("*.[jJ][pP][gG]")) + list(img_dir.glob("*.[pP][nN][gG]"))
                                for img_file in img_files[:5]:
                                    image_text += await self._transcribe_image(img_file)
                        if image_text:
                            text += "\n\n=== IMAGE DATA ===\n" + image_text
                    
                    await rag.ainsert(text)
                    results[filepath.name] = True
                    print(f"    ✓ Indexed successfully")
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
            "embedding_type": self.EMBEDDING_MODELS[self.embedding_model]["type"],
            "embedding_dim": self.EMBEDDING_MODELS[self.embedding_model]["dim"],
            "available": LIGHTRAG_AVAILABLE
        }
    
    def delete_database(self) -> None:
        """Delete all data and reset the database."""
        import shutil
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self._rag = None


if __name__ == "__main__":
    print("LightRAG Multimodal Engine")
    print(f"  LightRAG available: {LIGHTRAG_AVAILABLE}")
    print(f"  Local embeddings available: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    print(f"\nSupported embedding models:")
    for name, info in LightRAGEngine.EMBEDDING_MODELS.items():
        print(f"  - {name}: {info['dim']}d ({info['type']})")
    print(f"\nConfiguration from .env:")
    print(f"  LLM_BASE_URL: {os.getenv('LLM_BASE_URL', '(not set)')}")
    print(f"  LLM_MODEL: {os.getenv('LLM_MODEL', '(not set)')}")
    print(f"  EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', '(not set)')}")
