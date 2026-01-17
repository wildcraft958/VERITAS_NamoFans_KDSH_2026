"""
KDSH 2026: Constraint-Driven Narrative Auditor

A 5-layer pipeline for narrative consistency verification:
- Layer 1: Atomic Constraint Decomposition (FActScore + Zero-Shot)
- Layer 2: Dual Retrieval Indexing (RAPTOR + HippoRAG)
- Layer 3: Parallel Verification Streams (Historian + Simulator)
- Layer 4: Evidence Ledger Aggregation
- Layer 5: Conservative Adjudication Rules
"""

__version__ = "1.0.0"
__author__ = "Animesh Raj"
