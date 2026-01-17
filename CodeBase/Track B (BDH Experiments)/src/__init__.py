"""
BDH Source Package
==================

Main package for BDH Narrative Consistency Detection.

Structure:
- src.models: BDH architecture (from official repo)
- src.signals: Individual consistency signals
- src.pipeline: Main processing pipeline
- src.utils: Utilities (visualization, coreference)
"""

# Re-export main interfaces
from src.pipeline import (
    SOTANarrativeConsistencyPipeline,
    create_sota_pipeline,
)

from src.signals import (
    BDHScanner,
    DocumentNLIClassifier,
    TemporalNarrativeReasoner,
)

__all__ = [
    'SOTANarrativeConsistencyPipeline',
    'create_sota_pipeline',
    'BDHScanner',
    'DocumentNLIClassifier',
    'TemporalNarrativeReasoner',
]
