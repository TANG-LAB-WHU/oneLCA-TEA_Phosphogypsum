"""
LightRAG Engine Module

Graph-enhanced Retrieval Augmented Generation engine for phosphogypsum literature.
Uses LightRAG for entity-relationship extraction and knowledge graph-based retrieval.
"""

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from dotenv import load_dotenv
import PIL.Image

# Load environment variables
load_dotenv()

try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
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
    mode: str  # local, global, hybrid, mix


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
    - Support for Gemini/OpenAI LLMs
    """
    
    def __init__(
        self,
        working_dir: Optional[Path] = None,
        llm_model: str = "gemini-2.0-flash",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the LightRAG engine.
        
        Args:
            working_dir: Directory for storing graph and vector data
            llm_model: LLM model for entity extraction and generation
            embedding_model: Embedding model for vector search
        """
        if not LIGHTRAG_AVAILABLE:
            raise ImportError(
                "lightrag not installed. Run: pip install lightrag-hku"
            )
        
        self.working_dir = working_dir or Path("./data/processed/lightrag_db")
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Get API keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize LightRAG
        self._rag = None
    
    def _get_rag(self) -> LightRAG:
        """Lazy initialize LightRAG instance."""
        if self._rag is None:
            # Configure LLM function based on available API
            if self.gemini_api_key:
                llm_func = self._create_gemini_llm_func()
                embed_func = self._create_gemini_embed_func()
            elif self.openai_api_key:
                llm_func = self._create_openai_llm_func()
                embed_func = self._create_openai_embed_func()
            else:
                raise ValueError(
                    "No API key found. Set GEMINI_API_KEY or OPENAI_API_KEY in .env"
                )
            
            self._rag = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=llm_func,
                embedding_func=embed_func
            )
        
        return self._rag
    
    def _create_gemini_llm_func(self):
        """Create LLM function for Gemini API."""
        import google.generativeai as genai
        
        genai.configure(api_key=self.gemini_api_key)
        
        async def llm_func(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            **kwargs
        ) -> str:
            model = genai.GenerativeModel(self.llm_model)
            
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = model.generate_content(full_prompt)
            return response.text
        
        return llm_func
    
    def _create_gemini_embed_func(self) -> EmbeddingFunc:
        """Create embedding function for Gemini API."""
        import google.generativeai as genai
        import numpy as np
        
        genai.configure(api_key=self.gemini_api_key)
        
        async def embed_func(texts: list[str]) -> np.ndarray:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text
                )
                embeddings.append(result['embedding'])
            return np.array(embeddings)
        
        return EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=embed_func
        )
    
    def _create_openai_llm_func(self):
        """Create LLM function for OpenAI API."""
        async def llm_func(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            **kwargs
        ) -> str:
            return await openai_complete_if_cache(
                self.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=self.openai_api_key,
                **kwargs
            )
        return llm_func
    
    def _create_openai_embed_func(self) -> EmbeddingFunc:
        """Create embedding function for OpenAI API."""
        return EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model=self.embedding_model,
                api_key=self.openai_api_key
            )
        )
    
    def add_document(
        self,
        text: str,
        document_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a document to the LightRAG database.
        
        Args:
            text: Document text content
            document_id: Optional unique identifier
            metadata: Optional document metadata
        """
        rag = self._get_rag()
        
        # LightRAG handles chunking and entity extraction internally
        asyncio.run(rag.ainsert(text))
    
    def add_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.md",
        limit: int = None,
        transcribe_images: bool = True
    ) -> Dict[str, bool]:
        """
        Add all documents from a directory, including image transcriptions.
        
        Args:
            directory: Directory containing documents
            pattern: Glob pattern for files
            limit: Maximum number of documents to process
            transcribe_images: Whether to search for and transcribe associated images
            
        Returns:
            Dict mapping filename to success status
        """
        directory = Path(directory)
        results = {}
        
        # Support MinerU output structure (files can be in subdirectories)
        files = list(directory.rglob(pattern))
        if limit:
            files = files[:limit]
        
        rag = self._get_rag()
        
        for i, filepath in enumerate(files, 1):
            try:
                print(f"[{i}/{len(files)}] Processing: {filepath.name[:50]}...")
                
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Check for associated images (MinerU style)
                if transcribe_images:
                    # Check in same dir or parent dir's 'images' folder
                    image_dirs = [
                        filepath.parent / "images",
                        filepath.parent.parent / "images"
                    ]
                    
                    image_text = ""
                    for img_dir in image_dirs:
                        if img_dir.exists() and img_dir.is_dir():
                            print(f"    - Found image directory: {img_dir.name}")
                            # Transcribe up to 10 images per document to save quota
                            img_files = list(img_dir.glob("*.[jJ][pP][gG]")) + \
                                        list(img_dir.glob("*.[pP][nN][gG]"))
                            
                            for img_file in img_files[:10]:
                                print(f"      * Transcribing: {img_file.name}")
                                desc = asyncio.run(self._transcribe_image(img_file))
                                image_text += desc
                    
                    if image_text:
                        text += "\n\n" + "="*30 + "\nIMAGE DATA ENHANCEMENT\n" + "="*30 + "\n"
                        text += image_text
                
                # Insert document with multimodal enhancement
                asyncio.run(rag.ainsert(text))
                results[filepath.name] = True
                print(f"    ✓ Indexed successfully with multimodal support")
                
            except Exception as e:
                results[filepath.name] = False
                print(f"    ✗ Error: {e}")
        
        return results
    
    def query(
        self,
        query: str,
        mode: str = "mix"
    ) -> QueryResult:
        """
        Query the knowledge base.
        
        Args:
            query: User query
            mode: Query mode - 'local', 'global', 'hybrid', or 'mix'
                - local: Uses entity relationships for specific facts
                - global: Uses high-level summaries for themes
                - hybrid: Combines local and global
                - mix: Dynamically selects based on query (recommended)
                
        Returns:
            QueryResult with answer and metadata
        """
        rag = self._get_rag()
        
        # Run query
        answer = asyncio.run(rag.aquery(query, param=QueryParam(mode=mode)))
        
        return QueryResult(
            query=query,
            answer=answer,
            mode=mode
        )
    
    def retrieve(
        self,
        query: str,
        mode: str = "mix"
    ) -> RetrievalResult:
        """
        Retrieve relevant information (alias for query).
        
        Provided for API compatibility with previous LightRAGEngine.
        """
        result = self.query(query, mode)
        return RetrievalResult(
            query=result.query,
            answer=result.answer,
            mode=result.mode
        )
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        return {
            "working_dir": str(self.working_dir),
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "available": LIGHTRAG_AVAILABLE
        }
    
    def delete_database(self) -> None:
        """Delete all data and reset the database."""
        import shutil
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
        self._rag = None



if __name__ == "__main__":
    print("LightRAG Engine Module")
    print("-" * 40)
    print(f"LightRAG available: {LIGHTRAG_AVAILABLE}")
    
    if LIGHTRAG_AVAILABLE:
        print("\nUsage example:")
        print("  from pgloop.knowledge import LightRAGEngine")
        print("  rag = LightRAGEngine()")
        print("  rag.add_document('Your document text...')")
        print("  result = rag.query('Your question?')")
        print("  print(result.answer)")
