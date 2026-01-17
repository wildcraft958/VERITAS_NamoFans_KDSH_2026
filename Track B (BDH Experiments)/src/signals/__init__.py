"""
BDH Signals - Consistency Detection Components
==============================================

Individual signal extractors for narrative consistency:
- bdh_scanner: Hebbian Learning (Synaptic Drift)
- nli_classifier: DeBERTa NLI
- temporal_graph: Allen's Interval Algebra
- causal_extractor: Constraint Graph
- cross_encoder_evidence: Evidence Attribution
- neural_event_extractor: SpaCy + BERT-NER
"""

from src.signals.bdh_scanner import BDHScanner, ConsistencyScore, sparsify
from src.signals.nli_classifier import DocumentNLIClassifier, DocumentNLIResult
from src.signals.temporal_graph import TemporalNarrativeReasoner, TemporalGraph
from src.signals.causal_extractor import CausalEventExtractor, ConstraintViolationDetector
from src.signals.cross_encoder_evidence import CrossEncoderEvidenceAttributor
from src.signals.neural_event_extractor import NeuralEventExtractor
from src.signals.evidence_attributor import EvidenceRetriever

__all__ = [
    'BDHScanner',
    'ConsistencyScore',
    'sparsify',
    'DocumentNLIClassifier',
    'DocumentNLIResult',
    'TemporalNarrativeReasoner',
    'TemporalGraph',
    'CausalEventExtractor',
    'ConstraintViolationDetector',
    'CrossEncoderEvidenceAttributor',
    'NeuralEventExtractor',
    'EvidenceRetriever',
]
