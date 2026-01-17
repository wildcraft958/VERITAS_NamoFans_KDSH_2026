"""
Pathway Integration Module
==========================

Optional Pathway framework integration for enhanced document ingestion
and vector store retrieval.

This module is OPTIONAL - all functionality works without Pathway installed.
When available, Pathway provides:
1. Real-time document connectors (Google Drive, local folders)
2. Vector store with automatic index updates  
3. Streaming retrieval for evidence attribution

Usage:
    from src.pathway_integration import PATHWAY_AVAILABLE
    
    if PATHWAY_AVAILABLE:
        from src.pathway_integration import PathwayDocumentConnector
        connector = PathwayDocumentConnector.from_local("/path/to/books")
"""

# Feature detection
PATHWAY_AVAILABLE = False

try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
except ImportError:
    pw = None

# Conditional exports
__all__ = ['PATHWAY_AVAILABLE']

if PATHWAY_AVAILABLE:
    from .connector import PathwayDocumentConnector
    from .vector_store import PathwayVectorStore
    from .retriever import PathwayRetriever
    
    __all__.extend([
        'PathwayDocumentConnector',
        'PathwayVectorStore', 
        'PathwayRetriever',
    ])
