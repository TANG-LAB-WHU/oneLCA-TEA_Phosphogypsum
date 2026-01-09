"""
RAG Engine Module

Retrieval Augmented Generation engine for querying phosphogypsum literature.
Uses ChromaDB for vector storage and supports multiple embedding models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@dataclass
class RetrievalResult:
    """Result from RAG retrieval."""
    
    query: str
    documents: List[str]
    metadatas: List[Dict]
    distances: List[float]
    ids: List[str]


@dataclass
class GenerationResult:
    """Result from RAG generation."""
    
    query: str
    answer: str
    sources: List[Dict]
    confidence: float


class RAGEngine:
    """
    Retrieval Augmented Generation engine for phosphogypsum literature.
    
    Features:
    - Document ingestion and chunking
    - Vector embedding and storage
    - Semantic search
    - LLM-augmented answer generation
    """
    
    def __init__(
        self,
        collection_name: str = "phosphogypsum_lit",
        persist_directory: Optional[Path] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the RAG engine.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Run: pip install chromadb")
        
        self.persist_directory = persist_directory or Path("./data/processed/chroma_db")
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.embedding_model_name = embedding_model
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._embedding_model
    
    def add_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict[str, Any] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> int:
        """
        Add a document to the RAG database.
        
        Args:
            document_id: Unique identifier for the document
            text: Document text content
            metadata: Document metadata (title, source, etc.)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            Number of chunks added
        """
        # Split text into chunks
        chunks = self._split_text(text, chunk_size, chunk_overlap)
        
        # Prepare data for ChromaDB
        ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "document_id": document_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        
        # Add to collection
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        return len(chunks)
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind(". ")
                if last_period > chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_documents_from_directory(
        self,
        directory: Union[str, Path],
        pattern: str = "*.md",
        metadata_extractor: callable = None
    ) -> Dict[str, int]:
        """
        Add all documents from a directory.
        
        Args:
            directory: Directory containing documents
            pattern: Glob pattern for files
            metadata_extractor: Function to extract metadata from filepath
            
        Returns:
            Dict mapping document ID to chunk count
        """
        directory = Path(directory)
        results = {}
        
        for filepath in directory.glob(pattern):
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            
            doc_id = filepath.stem
            metadata = {"filename": filepath.name, "path": str(filepath)}
            
            if metadata_extractor:
                metadata.update(metadata_extractor(filepath))
            
            chunks_added = self.add_document(doc_id, text, metadata)
            results[doc_id] = chunks_added
        
        return results
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None,
        where_document: Dict = None
    ) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            RetrievalResult with matched documents
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return RetrievalResult(
            query=query,
            documents=results["documents"][0] if results["documents"] else [],
            metadatas=results["metadatas"][0] if results["metadatas"] else [],
            distances=results["distances"][0] if results["distances"] else [],
            ids=results["ids"][0] if results["ids"] else []
        )
    
    def query_with_generation(
        self,
        query: str,
        llm_extractor,  # LLMExtractor instance
        n_results: int = 5,
        system_prompt: str = None
    ) -> GenerationResult:
        """
        Query with LLM-augmented answer generation.
        
        Args:
            query: User query
            llm_extractor: LLMExtractor instance for generation
            n_results: Number of documents to retrieve
            system_prompt: Custom system prompt
            
        Returns:
            GenerationResult with answer and sources
        """
        # Retrieve relevant documents
        retrieval = self.retrieve(query, n_results)
        
        if not retrieval.documents:
            return GenerationResult(
                query=query,
                answer="No relevant documents found.",
                sources=[],
                confidence=0.0
            )
        
        # Build context from retrieved documents
        context = "\n\n---\n\n".join([
            f"[Source {i+1}]\n{doc}"
            for i, doc in enumerate(retrieval.documents)
        ])
        
        # Generate answer using LLM
        prompt = f"""Based on the following context, answer the question.
If the answer cannot be found in the context, say so.
Always cite your sources using [Source N] notation.

Context:
{context}

Question: {query}

Answer:"""
        
        result = llm_extractor.extract(
            prompt,
            "composition",  # Use any extraction type, we just need the raw response
            custom_prompt=prompt
        )
        
        # Extract sources
        sources = [
            {
                "id": retrieval.ids[i],
                "metadata": retrieval.metadatas[i],
                "distance": retrieval.distances[i]
            }
            for i in range(len(retrieval.documents))
        ]
        
        return GenerationResult(
            query=query,
            answer=result.raw_response,
            sources=sources,
            confidence=1.0 - (sum(retrieval.distances) / len(retrieval.distances)) if retrieval.distances else 0.0
        )
    
    def get_statistics(self) -> Dict:
        """Get collection statistics."""
        return {
            "collection_name": self.collection.name,
            "document_count": self.collection.count(),
            "persist_directory": str(self.persist_directory)
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection.name)


if __name__ == "__main__":
    # Example usage
    rag = RAGEngine()
    
    # Add a sample document
    sample_doc = """
    Phosphogypsum is a waste byproduct from phosphoric acid production.
    It contains calcium sulfate dihydrate and various impurities including
    heavy metals and radionuclides. Common treatment methods include
    stack disposal, use in cement production, and agricultural application.
    """
    
    rag.add_document(
        "sample_doc_001",
        sample_doc,
        {"title": "Phosphogypsum Overview", "year": 2024}
    )
    
    # Query
    results = rag.retrieve("What are phosphogypsum treatment methods?")
    print(f"Found {len(results.documents)} relevant chunks")
