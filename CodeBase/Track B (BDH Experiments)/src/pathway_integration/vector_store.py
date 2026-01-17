"""
Pathway Vector Store
====================

Vector store implementation using Pathway's indexing infrastructure.

Features:
- Real-time index updates when documents change
- BruteForce KNN for accurate similarity search
- Integration with sentence-transformers for embeddings

This provides an alternative to the manual chunking in CrossEncoderEvidenceAttributor.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os

try:
    import pathway as pw
    from pathway.stdlib.indexing import BruteForceKnnFactory
    PATHWAY_AVAILABLE = True
except ImportError:
    pw = None
    BruteForceKnnFactory = None
    PATHWAY_AVAILABLE = False


@dataclass
class VectorSearchResult:
    """Result from vector similarity search."""
    text: str
    score: float
    chunk_index: int
    metadata: Dict[str, Any]


class PathwayVectorStore:
    """
    Vector store using Pathway's real-time indexing.
    
    Uses BruteForceKNN for accurate similarity search with
    automatic index updates when source documents change.
    
    Example:
        store = PathwayVectorStore()
        store.index_document(novel_text)
        results = store.query("character born in London", top_k=5)
    """
    
    def __init__(
        self,
        embedder_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,  # words per chunk
        chunk_overlap: int = 50,
    ):
        """
        Initialize vector store.
        
        Args:
            embedder_name: Sentence-transformer model name
            chunk_size: Words per chunk
            chunk_overlap: Overlap between chunks
        """
        if not PATHWAY_AVAILABLE:
            raise ImportError(
                "Pathway is not installed. Install with: pip install pathway-python"
            )
        
        self.embedder_name = embedder_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Lazy load embedder
        self._embedder = None
        self._index = None
        self._chunks: List[str] = []
        
    def _get_embedder(self):
        """Lazy load the embedder."""
        if self._embedder is None:
            try:
                # Try Pathway's built-in embedder first
                from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
                self._embedder = SentenceTransformerEmbedder(
                    model=self.embedder_name
                )
            except ImportError:
                # Fallback to direct sentence-transformers
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedder_name)
        return self._embedder
    
    def _chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.
        
        Returns:
            List of (chunk_text, char_start, char_end)
        """
        words = text.split()
        chunks = []
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions (approximate)
            char_start = len(' '.join(words[:i]))
            char_end = char_start + len(chunk_text)
            
            chunks.append((chunk_text, char_start, char_end))
            
            # Move by (chunk_size - overlap)
            i += self.chunk_size - self.chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def index_document(self, text: str, doc_id: str = "novel") -> int:
        """
        Index a document into the vector store.
        
        Args:
            text: Full document text
            doc_id: Document identifier
            
        Returns:
            Number of chunks indexed
        """
        # Chunk the text
        chunks = self._chunk_text(text)
        self._chunks = [c[0] for c in chunks]
        
        # Create Pathway table from chunks
        chunk_data = [
            {
                "text": chunk_text,
                "chunk_index": i,
                "char_start": char_start,
                "char_end": char_end,
                "doc_id": doc_id,
            }
            for i, (chunk_text, char_start, char_end) in enumerate(chunks)
        ]
        
        # Build table
        table = pw.debug.table_from_rows(
            pw.schema_builder()
                .column("text", pw.Type.STRING)
                .column("chunk_index", pw.Type.INT)
                .column("char_start", pw.Type.INT)
                .column("char_end", pw.Type.INT)
                .column("doc_id", pw.Type.STRING)
                .build(),
            [tuple(d.values()) for d in chunk_data]
        )
        
        # Build KNN index
        embedder = self._get_embedder()
        
        self._index = BruteForceKnnFactory(
            embedder=embedder,
        ).build_index(table, column=table.text)
        
        return len(chunks)
    
    def query(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> List[VectorSearchResult]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            
        Returns:
            List of VectorSearchResult objects
        """
        if self._index is None:
            return []
        
        # Query the index
        results = self._index.query(
            query_text,
            k=top_k,
        )
        
        # Convert to result objects
        search_results = []
        for i, (text, score) in enumerate(results):
            search_results.append(VectorSearchResult(
                text=text,
                score=float(score),
                chunk_index=i,  # Note: actual index lost in query
                metadata={},
            ))
        
        return search_results
    
    def get_chunks(self) -> List[str]:
        """Get all indexed chunks."""
        return self._chunks.copy()
    
    def clear(self):
        """Clear the index."""
        self._index = None
        self._chunks = []


class PathwayVectorStoreSimple:
    """
    Simplified vector store that works without full Pathway setup.
    
    Uses sentence-transformers directly for embedding and
    numpy for cosine similarity search.
    
    This is the recommended fallback when Pathway's streaming
    features are not needed.
    """
    
    def __init__(
        self,
        embedder_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        self.embedder_name = embedder_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self._model = None
        self._chunks: List[str] = []
        self._embeddings = None
    
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedder_name)
        return self._model
    
    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            i += self.chunk_size - self.chunk_overlap
        return chunks
    
    def index_document(self, text: str) -> int:
        import numpy as np
        
        self._chunks = self._chunk_text(text)
        model = self._get_model()
        self._embeddings = model.encode(self._chunks, convert_to_numpy=True)
        
        return len(self._chunks)
    
    def query(self, query_text: str, top_k: int = 5) -> List[VectorSearchResult]:
        import numpy as np
        
        if self._embeddings is None or len(self._chunks) == 0:
            return []
        
        model = self._get_model()
        query_emb = model.encode([query_text], convert_to_numpy=True)[0]
        
        # Cosine similarity
        similarities = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append(VectorSearchResult(
                text=self._chunks[idx],
                score=float(similarities[idx]),
                chunk_index=int(idx),
                metadata={},
            ))
        
        return results
