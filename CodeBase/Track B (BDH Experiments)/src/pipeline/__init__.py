"""
BDH Pipeline - Core Processing Logic
====================================

Main pipeline components:
- sota_pipeline: Main SOTA Narrative Consistency Pipeline
- backstory_parser: Text parsing and claim extraction
- chunk_aggregator: Text chunking utilities
"""

from src.pipeline.sota_pipeline import (
    SOTANarrativeConsistencyPipeline,
    SOTAConsistencyVerdict,
    create_sota_pipeline
)
from src.pipeline.backstory_parser import (
    SemanticBackstoryParser,
    SemanticBackstoryEmbedder,
    DualStateMemory
)
from src.pipeline.chunk_aggregator import ChunkAggregator, HierarchicalAggregator

__all__ = [
    'SOTANarrativeConsistencyPipeline',
    'SOTAConsistencyVerdict',
    'create_sota_pipeline',
    'SemanticBackstoryParser',
    'SemanticBackstoryEmbedder',
    'DualStateMemory',
    'ChunkAggregator',
    'HierarchicalAggregator',
]
