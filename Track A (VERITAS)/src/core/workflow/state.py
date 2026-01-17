"""
LangGraph state definition for the Narrative Auditor workflow.
"""

from typing import TypedDict, Annotated, Sequence, Any

from core.layer1.constraint_graph import Constraint, ConstraintGraph
from core.layer2.unified_retriever import RetrievalResult
from core.layer3.historian import HistorianVerdict
from core.layer3.simulator import SimulatorVerdict, Persona
from core.layer4.ledger import LedgerEntry
from core.layer5.adjudicator import FinalVerdict


class NarrativeState(TypedDict, total=False):
    """
    State for the Narrative Auditor LangGraph workflow.
    
    This state flows through all 5 layers, accumulating results
    at each step.
    """
    
    # Input data
    backstory_id: str
    backstory: str
    novel_text: str
    novel_id: str
    
    # Layer 1: Atomic Decomposition
    constraint_graph: ConstraintGraph
    constraints: list[Constraint]
    
    # Layer 2: Dual Retrieval
    retrieval_results: dict[str, RetrievalResult]
    
    # Layer 3: Parallel Verification
    persona: Persona
    historian_verdicts: dict[str, HistorianVerdict]
    simulator_verdicts: dict[str, SimulatorVerdict]
    
    # Layer 4: Evidence Ledger
    ledger_entries: list[LedgerEntry]
    
    # Layer 5: Final Verdict
    final_verdict: FinalVerdict
    
    # Output
    prediction: int  # 0 or 1
    confidence: float
    reasoning: list[str]
    
    # Metadata
    error: str | None
    current_layer: str


def create_initial_state(
    backstory: str,
    novel_text: str,
    backstory_id: str = "default",
    novel_id: str = "default"
) -> NarrativeState:
    """
    Create initial state for the workflow.
    
    Args:
        backstory: Character backstory text
        novel_text: Full novel text
        backstory_id: Unique ID for the backstory
        novel_id: Unique ID for the novel
        
    Returns:
        Initial NarrativeState
    """
    return {
        "backstory_id": backstory_id,
        "backstory": backstory,
        "novel_text": novel_text,
        "novel_id": novel_id,
        "constraints": [],
        "retrieval_results": {},
        "historian_verdicts": {},
        "simulator_verdicts": {},
        "ledger_entries": [],
        "prediction": 0,
        "confidence": 0.0,
        "reasoning": [],
        "error": None,
        "current_layer": "init"
    }
