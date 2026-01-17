"""
Constraint data structures for Layer 1.

Defines atomic facts and constraints with their types.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.layer1.factscore import FactExtractor
    from core.layer1.classifier import ConstraintClassifier


class ConstraintType(str, Enum):
    """Types of constraints that can be extracted from backstories."""
    
    TEMPORAL = "Temporal"           # Age, timeline, sequence of events
    PSYCHOLOGICAL = "Psychological"  # Traits, skills, beliefs, fears
    CAUSAL = "Causal"               # Events, actions, consequences
    WORLD_MODEL = "World-Model"     # Assumptions about story world rules


@dataclass
class AtomicFact:
    """
    A single, independent, testable fact extracted from a backstory.
    
    Atomic facts are the smallest units of information that can be
    verified against the novel independently.
    """
    
    id: str
    text: str
    source_sentence: str = ""
    
    def __str__(self) -> str:
        return self.text


@dataclass
class Constraint:
    """
    A constraint derived from one or more atomic facts.
    
    Constraints are the logical claims that must hold if the
    backstory is consistent with the novel.
    """
    
    id: str
    type: ConstraintType
    claim: str
    source_atoms: list[AtomicFact] = field(default_factory=list)
    expected_in_novel: bool = True
    verification_methods: list[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def __post_init__(self):
        """Set default verification methods based on constraint type."""
        if not self.verification_methods:
            if self.type == ConstraintType.TEMPORAL:
                self.verification_methods = ["HIPPORAG_TEMPORAL_EDGES", "RAPTOR_TIMELINE"]
            elif self.type == ConstraintType.PSYCHOLOGICAL:
                self.verification_methods = ["RAPTOR_CHARACTER_ARC", "SIMULATOR_PERSONA"]
            elif self.type == ConstraintType.CAUSAL:
                self.verification_methods = ["HIPPORAG_CAUSAL_PATHS", "RAPTOR_NARRATIVE_ARC"]
            elif self.type == ConstraintType.WORLD_MODEL:
                self.verification_methods = ["RAPTOR_THEMATIC", "HISTORIAN_EVIDENCE"]


@dataclass
class ConstraintGraph:
    """
    Collection of constraints extracted from a backstory.
    
    Provides methods for querying and iterating over constraints.
    """
    
    backstory_id: str
    constraints: list[Constraint] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.constraints)
    
    def __iter__(self):
        return iter(self.constraints)
    
    def get_by_type(self, constraint_type: ConstraintType) -> list[Constraint]:
        """Get all constraints of a specific type."""
        return [c for c in self.constraints if c.type == constraint_type]
    
    def get_high_priority(self) -> list[Constraint]:
        """Get constraints with causal or temporal types (higher priority for verification)."""
        return [
            c for c in self.constraints 
            if c.type in (ConstraintType.CAUSAL, ConstraintType.TEMPORAL)
        ]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "backstory_id": self.backstory_id,
            "constraints": [
                {
                    "id": c.id,
                    "type": c.type.value,
                    "claim": c.claim,
                    "source_atoms": [{"id": a.id, "text": a.text} for a in c.source_atoms],
                    "expected_in_novel": c.expected_in_novel,
                    "verification_methods": c.verification_methods,
                    "confidence": c.confidence
                }
                for c in self.constraints
            ]
        }


def decompose_backstory(
    backstory: str,
    backstory_id: str = "default",
    fact_extractor: "FactExtractor | None" = None,
    classifier: "ConstraintClassifier | None" = None
) -> ConstraintGraph:
    """
    Full Layer 1 processing: decompose backstory into constraint graph.
    
    This is the main entry point for Layer 1. It:
    1. Extracts atomic facts using FActScore methodology
    2. Classifies each fact into constraint types
    3. Builds the constraint graph
    
    Args:
        backstory: The character backstory text
        backstory_id: Unique identifier for this backstory
        fact_extractor: Optional custom fact extractor
        classifier: Optional custom classifier
        
    Returns:
        ConstraintGraph with all extracted constraints
    """
    # Import here to avoid circular imports
    from core.layer1.factscore import FactExtractor
    from core.layer1.classifier import ConstraintClassifier
    
    extractor = fact_extractor or FactExtractor()
    clf = classifier or ConstraintClassifier()
    
    # Step 1: Extract atomic facts
    atoms = extractor.extract(backstory)
    
    # Step 2: Classify and build constraints
    constraints = []
    for i, atom in enumerate(atoms):
        constraint_type, confidence = clf.classify(atom)
        
        constraint = Constraint(
            id=f"C{i+1}_{backstory_id}",
            type=constraint_type,
            claim=atom.text,
            source_atoms=[atom],
            confidence=confidence
        )
        constraints.append(constraint)
    
    return ConstraintGraph(backstory_id=backstory_id, constraints=constraints)
