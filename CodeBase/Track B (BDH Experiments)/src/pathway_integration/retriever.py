"""
Pathway Retriever
=================

High-level retriever that combines document ingestion and vector search.

This is the main interface for integrating Pathway with the SOTA pipeline.
It provides a drop-in replacement for the manual chunking approach.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    pw = None
    PATHWAY_AVAILABLE = False


@dataclass
class RetrievedPassage:
    """A passage retrieved from the narrative."""
    text: str
    score: float
    chunk_index: int
    char_start: int = 0
    char_end: int = 0


class PathwayRetriever:
    """
    High-level retriever for narrative evidence.
    
    Combines:
    - Document ingestion (from connector or direct text)
    - Vector indexing (via PathwayVectorStore)
    - Semantic retrieval (query -> relevant passages)
    
    Example:
        retriever = PathwayRetriever()
        retriever.ingest("Once upon a time in a kingdom far away...")
        passages = retriever.retrieve("the princess lived in a castle", k=5)
    """
    
    def __init__(
        self,
        embedder_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        use_pathway: bool = True,
    ):
        """
        Initialize retriever.
        
        Args:
            embedder_name: Model for sentence embeddings
            chunk_size: Words per chunk
            chunk_overlap: Overlap between chunks
            use_pathway: Use full Pathway (vs simplified numpy backend)
        """
        self.embedder_name = embedder_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pathway = use_pathway and PATHWAY_AVAILABLE
        
        self._store = None
        self._chunks: List[str] = []
        self._is_indexed = False
        
    def _init_store(self):
        """Initialize the vector store."""
        if self.use_pathway:
            from .vector_store import PathwayVectorStore
            self._store = PathwayVectorStore(
                embedder_name=self.embedder_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            from .vector_store import PathwayVectorStoreSimple
            self._store = PathwayVectorStoreSimple(
                embedder_name=self.embedder_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
    
    def ingest(self, novel_text: str) -> int:
        """
        Ingest a novel for retrieval.
        
        Args:
            novel_text: Full text of the novel
            
        Returns:
            Number of chunks indexed
        """
        if self._store is None:
            self._init_store()
        
        num_chunks = self._store.index_document(novel_text)
        self._chunks = self._store.get_chunks()
        self._is_indexed = True
        
        return num_chunks
    
    def retrieve(
        self, 
        query: str, 
        k: int = 5,
        rerank: bool = False,
    ) -> List[RetrievedPassage]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: Query text (e.g., a backstory claim)
            k: Number of passages to retrieve
            rerank: Apply cross-encoder reranking (slower, more accurate)
            
        Returns:
            List of RetrievedPassage objects
        """
        if not self._is_indexed:
            return []
        
        # Vector search
        results = self._store.query(query, top_k=k)
        
        # Convert to passages
        passages = []
        for r in results:
            passages.append(RetrievedPassage(
                text=r.text,
                score=r.score,
                chunk_index=r.chunk_index,
            ))
        
        # Optional reranking with cross-encoder
        if rerank and len(passages) > 0:
            passages = self._rerank(query, passages)
        
        return passages
    
    def _rerank(
        self, 
        query: str, 
        passages: List[RetrievedPassage]
    ) -> List[RetrievedPassage]:
        """Rerank passages using cross-encoder."""
        try:
            from sentence_transformers import CrossEncoder
            
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            
            pairs = [(query, p.text) for p in passages]
            scores = model.predict(pairs)
            
            for i, score in enumerate(scores):
                passages[i].score = float(score)
            
            passages.sort(key=lambda x: x.score, reverse=True)
            
        except ImportError:
            pass  # Keep original scores
        
        return passages
    
    def get_chunks(self) -> List[str]:
        """Get all indexed chunks."""
        return self._chunks.copy()
    
    @property
    def is_indexed(self) -> bool:
        """Check if a document has been indexed."""
        return self._is_indexed
    
    def clear(self):
        """Clear the index."""
        if self._store:
            self._store.clear()
        self._chunks = []
        self._is_indexed = False


def create_retriever(
    use_pathway: bool = False,
    **kwargs
) -> PathwayRetriever:
    """
    Factory function to create a retriever.
    
    Args:
        use_pathway: Enable Pathway features (if available)
        **kwargs: Additional arguments for PathwayRetriever
        
    Returns:
        Configured PathwayRetriever
    """
    return PathwayRetriever(use_pathway=use_pathway, **kwargs)
