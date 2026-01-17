"""
External Framework Integrations

This module provides adapters that directly use code from the cloned
external framework repositories (FActScore, RAPTOR, HippoRAG) instead
of abstracting them or using PyPI packages.

These integrations ensure that our pipeline uses the actual implementations
from the original research codebases.
"""

from core.external_integrations.factscore_adapter import (
    FActScoreAtomicFactGenerator,
    extract_atomic_facts
)
from core.external_integrations.raptor_adapter import (
    RAPTORTreeBuilder,
    build_raptor_tree,
    RAPTORTreeRetriever
)
from core.external_integrations.hipporag_adapter import (
    HippoRAGKnowledgeGraph,
    build_hipporag_index
)

__all__ = [
    # FActScore
    "FActScoreAtomicFactGenerator",
    "extract_atomic_facts",
    # RAPTOR
    "RAPTORTreeBuilder", 
    "build_raptor_tree",
    "RAPTORTreeRetriever",
    # HippoRAG
    "HippoRAGKnowledgeGraph",
    "build_hipporag_index",
]
