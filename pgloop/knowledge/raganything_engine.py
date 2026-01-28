"""
RAGAnything Engine Module

All-in-one multimodal RAG engine for phosphogypsum literature processing.
Provides unified document parsing, multimodal understanding, and knowledge graph indexing.

Built on top of LightRAG with integrated MinerU parsing pipeline.
"""

import os
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
from dotenv import load_dotenv

try:
    from raganything import RAGAnything, RAGAnythingConfig
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    RAGANYTHING_AVAILABLE = True
except ImportError:
    RAGANYTHING_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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
    
    Configuration via environment variables (.env):
        LLM_BASE_URL: OpenAI-compatible API base URL
        LLM_API_KEY: API key for LLM provider
        LLM_MODEL: LLM model name (default: gemini-3-flash)
        EMBEDDING_MODEL: Embedding model (default: bge-m3)
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        llm_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        parser: str = "mineru",
        parse_method: str = "auto"
    ):
        """
        Initialize RAGAnythingEngine.
        
        Args:
            working_dir: Directory for storing RAG data
            llm_model: LLM model name
            embedding_model: Embedding model name
            llm_base_url: LLM API base URL
            llm_api_key: LLM API key
            parser: Document parser ("mineru" or "docling")
            parse_method: Parse method ("auto", "ocr", "txt")
        """
        if not RAGANYTHING_AVAILABLE:
            raise ImportError(
                "RAGAnything not installed. Run: pip install 'raganything[all]'"
            )
        
        self.working_dir = Path(working_dir) if working_dir else Path("./data/processed/raganything_db")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # LLM configuration
        self.llm_base_url = llm_base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:8045/v1")
        self.llm_api_key = llm_api_key or os.getenv("LLM_API_KEY", "")
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gemini-3-flash")
        
        # Embedding configuration
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        
        # Parser configuration
        self.parser = parser
        self.parse_method = parse_method
        
        # Lazy-loaded instances
        self._rag = None
        self._local_embed_model = None
    
    def _get_local_embed_model(self):
        """Lazy load local embedding model."""
        if self._local_embed_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
            import torch
            print(f"Loading local embedding model: {self.embedding_model}")
            self._local_embed_model = SentenceTransformer(
                f"BAAI/{self.embedding_model}",
                device="cuda" if torch.cuda.is_available() else "cpu",
                local_files_only=True
            )
            device = "CUDA" if torch.cuda.is_available() else "CPU"
            print(f"Model loaded on: {device}")
        return self._local_embed_model
    
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
                **kwargs
            )
        return llm_model_func
    
    def _create_vision_func(self):
        """Create vision model function for multimodal processing."""
        llm_func = self._create_llm_func()
        
        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], 
            image_data=None, messages=None, **kwargs
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
                    **kwargs
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
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                },
                            ],
                        } if image_data else {"role": "user", "content": prompt},
                    ],
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url,
                    **kwargs
                )
            # Pure text
            else:
                return llm_func(prompt, system_prompt, history_messages, **kwargs)
        
        return vision_model_func
    
    def _create_embedding_func(self):
        """Create embedding function."""
        if self.embedding_model == "bge-m3" and SENTENCE_TRANSFORMERS_AVAILABLE:
            # Local embedding
            model = self._get_local_embed_model()
            return EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=lambda texts: model.encode(texts, normalize_embeddings=True).tolist()
            )
        else:
            # OpenAI-compatible API embedding
            return EmbeddingFunc(
                embedding_dim=3072,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model="text-embedding-3-large",
                    api_key=self.llm_api_key,
                    base_url=self.llm_base_url
                )
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
                enable_equation_processing=True
            )
            
            self._rag = RAGAnything(
                config=config,
                llm_model_func=self._create_llm_func(),
                vision_model_func=self._create_vision_func(),
                embedding_func=self._create_embedding_func()
            )
        return self._rag
    
    def process_document(
        self,
        file_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a document with full multimodal extraction.
        
        Args:
            file_path: Path to document (PDF, Office, image)
            output_dir: Output directory for parsed content
            **kwargs: Additional parsing options
            
        Returns:
            Processing result dictionary
        """
        rag = self._get_rag_instance()
        file_path = Path(file_path)
        output_dir = Path(output_dir) if output_dir else self.working_dir / "parsed"
        
        async def _process():
            return await rag.process_document_complete(
                file_path=str(file_path),
                output_dir=str(output_dir),
                parse_method=self.parse_method,
                **kwargs
            )
        
        return asyncio.run(_process())
    
    def process_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.pdf",
        limit: Optional[int] = None,
        **kwargs
    ) -> Dict[str, bool]:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory containing documents
            pattern: Glob pattern for file matching
            limit: Maximum number of documents to process
            **kwargs: Additional processing options
            
        Returns:
            Dictionary mapping filenames to success status
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        if limit:
            files = files[:limit]
        
        results = {}
        for i, filepath in enumerate(files, 1):
            try:
                print(f"[{i}/{len(files)}] Processing: {filepath.name[:50]}...")
                self.process_document(filepath, **kwargs)
                results[filepath.name] = True
                print(f"    ✓ Processed successfully")
            except Exception as e:
                results[filepath.name] = False
                print(f"    ✗ Error: {e}")
        
        return results
    
    def query(self, query: str, mode: str = "hybrid") -> str:
        """
        Query the knowledge base (text-only).
        
        Args:
            query: Query string
            mode: Retrieval mode ("naive", "local", "global", "hybrid", "mix")
            
        Returns:
            Query answer
        """
        rag = self._get_rag_instance()
        
        async def _query():
            return await rag.aquery(query, mode=mode)
        
        return asyncio.run(_query())
    
    def multimodal_query(
        self,
        query: str,
        multimodal_content: Optional[List[Dict]] = None,
        mode: str = "hybrid"
    ) -> MultimodalQueryResult:
        """
        Query with multimodal context.
        
        Args:
            query: Query string
            multimodal_content: List of multimodal content dicts, e.g.:
                [{"type": "equation", "latex": "E=mc^2", "equation_caption": "Einstein's formula"}]
                [{"type": "table", "content": "<table>...</table>", "table_caption": "Data table"}]
            mode: Retrieval mode
            
        Returns:
            MultimodalQueryResult with answer and context
        """
        rag = self._get_rag_instance()
        
        async def _query():
            if multimodal_content:
                result = await rag.aquery_with_multimodal(
                    query,
                    multimodal_content=multimodal_content,
                    mode=mode
                )
            else:
                result = await rag.aquery(query, mode=mode)
            return result
        
        answer = asyncio.run(_query())
        return MultimodalQueryResult(
            query=query,
            answer=answer,
            mode=mode,
            multimodal_context=multimodal_content
        )
    
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
            "embedding_model": self.embedding_model
        }


if __name__ == "__main__":
    print("RAGAnything Engine Module")
    print("-" * 40)
    print(f"RAGAnything available: {RAGANYTHING_AVAILABLE}")
    print(f"Sentence Transformers available: {SENTENCE_TRANSFORMERS_AVAILABLE}")
    
    if RAGANYTHING_AVAILABLE:
        print("\nQuick test:")
        engine = RAGAnythingEngine()
        print(f"Parser installed: {engine.check_parser_installation()}")
        print(f"Stats: {engine.get_statistics()}")
