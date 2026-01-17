"""
Unified dual retriever combining RAPTOR and HippoRAG results.
"""

from dataclasses import dataclass, field
from typing import Any

from core.layer1.constraint_graph import Constraint, ConstraintType


@dataclass
class Evidence:
    """A piece of evidence retrieved from the novel."""
    
    text: str
    source: str
    weight: float = 1.0
    level: int = 0  # RAPTOR level
    retriever: str = "raptor"  # "raptor" or "hipporag"
    metadata: dict = field(default_factory=dict)


@dataclass
class PathResult:
    """Result of a HippoRAG path query."""
    
    found: bool
    paths: list[tuple]  # List of (subject, relation, object) or multi-hop paths
    score: float
    
    def __bool__(self) -> bool:
        return self.found


@dataclass
class RetrievalResult:
    """Combined retrieval result from both RAPTOR and HippoRAG."""
    
    constraint_id: str
    raptor_evidence: list[Evidence] = field(default_factory=list)
    hipporag_paths: list[PathResult] = field(default_factory=list)
    
    @property
    def has_evidence(self) -> bool:
        """Check if any evidence was found."""
        return bool(self.raptor_evidence) or any(p.found for p in self.hipporag_paths)
    
    @property
    def top_evidence(self) -> Evidence | None:
        """Get highest-weighted evidence."""
        if not self.raptor_evidence:
            return None
        return max(self.raptor_evidence, key=lambda e: e.weight)
    
    def get_all_evidence_texts(self) -> list[str]:
        """Get all evidence texts."""
        texts = [e.text for e in self.raptor_evidence]
        for path_result in self.hipporag_paths:
            for path in path_result.paths:
                if isinstance(path, tuple) and len(path) >= 3:
                    texts.append(f"{path[0]} -> {path[1]} -> {path[2]}")
        return texts


class DualRetriever:
    """
    Unified interface for RAPTOR + HippoRAG dual retrieval.
    
    Combines hierarchical semantic search (RAPTOR) with
    knowledge graph path search (HippoRAG) for comprehensive
    evidence retrieval.
    """
    
    def __init__(
        self,
        novel_text: str | None = None,
        novel_id: str = "default"
    ):
        """
        Initialize dual retriever.
        
        Args:
            novel_text: Full novel text
            novel_id: Identifier for caching
        """
        from core.layer2.raptor_index import RAPTORIndex
        from core.layer2.hipporag_index import HippoRAGIndex
        
        self.novel_id = novel_id
        self.raptor: RAPTORIndex | None = None
        self.hipporag: HippoRAGIndex | None = None
        
        if novel_text:
            self.build(novel_text, novel_id)
    
    def build(self, novel_text: str, novel_id: str = "default") -> None:
        """Build both indexes from novel text."""
        from core.layer2.raptor_index import RAPTORIndex
        from core.layer2.hipporag_index import HippoRAGIndex
        
        self.novel_id = novel_id
        self.raptor = RAPTORIndex(novel_text)
        self.hipporag = HippoRAGIndex(novel_text)
    
    def retrieve(self, constraint: Constraint) -> RetrievalResult:
        """
        Retrieve evidence for a constraint from both indexes.
        
        Args:
            constraint: The constraint to find evidence for
            
        Returns:
            Combined RetrievalResult from both retrievers
        """
        result = RetrievalResult(constraint_id=constraint.id)
        
        # RAPTOR semantic search
        if self.raptor:
            raptor_results = self._query_raptor(constraint)
            result.raptor_evidence = raptor_results
        
        # HippoRAG path search
        if self.hipporag:
            hipporag_results = self._query_hipporag(constraint)
            result.hipporag_paths = hipporag_results
        
        return result
    
    def _query_raptor(self, constraint: Constraint) -> list[Evidence]:
        """Query RAPTOR index based on constraint type."""
        if not self.raptor:
            return []
        
        # Determine query strategy based on constraint type
        if constraint.type == ConstraintType.TEMPORAL:
            # Search at lower levels for specific timeline details
            min_level, max_level = 0, 1
        elif constraint.type == ConstraintType.PSYCHOLOGICAL:
            # Search at higher levels for character arcs
            min_level, max_level = 1, self.raptor.tree.max_level
        elif constraint.type == ConstraintType.CAUSAL:
            # Search across all levels for events
            min_level, max_level = 0, self.raptor.tree.max_level
        else:
            # Default: search all levels
            min_level, max_level = 0, self.raptor.tree.max_level
        
        # Query
        query = constraint.claim
        results = self.raptor.query(
            query=query,
            min_level=min_level,
            max_level=max_level
        )
        
        # Convert to Evidence objects
        evidence_list = []
        for node, score in results:
            evidence = Evidence(
                text=node.text,
                source=f"RAPTOR L{node.level} node {node.id}",
                weight=score,
                level=node.level,
                retriever="raptor",
                metadata=node.metadata
            )
            evidence_list.append(evidence)
        
        return evidence_list
    
    def _query_hipporag(self, constraint: Constraint) -> list[PathResult]:
        """Query HippoRAG index for relevant paths."""
        if not self.hipporag:
            return []
        
        results = []
        
        # Extract entities from constraint claim
        entities = self._extract_entities_from_claim(constraint.claim)
        
        # Query paths between entities
        for i, entity1 in enumerate(entities):
            # Query direct neighbors
            path_result = self.hipporag.query_path(subject=entity1)
            if path_result.found:
                results.append(path_result)
            
            # Query paths to other entities
            for entity2 in entities[i+1:]:
                path_result = self.hipporag.query_path(
                    subject=entity1,
                    object=entity2
                )
                if path_result.found:
                    results.append(path_result)
        
        return results
    
    def _extract_entities_from_claim(self, claim: str) -> list[str]:
        """Extract potential entity names from a claim."""
        # Simple extraction: capitalize words that might be names
        words = claim.split()
        entities = []
        
        for word in words:
            # Clean word
            clean = word.strip(".,!?\"'()[]")
            
            # Check if it looks like a name (capitalized, not at sentence start)
            if clean and clean[0].isupper() and len(clean) > 2:
                entities.append(clean)
        
        # Also add key nouns
        key_terms = ["father", "mother", "captivity", "escape", "death", 
                     "avalanche", "mountain", "skill", "knowledge"]
        for term in key_terms:
            if term in claim.lower():
                entities.append(term)
        
        return entities[:5]  # Limit to 5 entities
    
    def retrieve_batch(
        self, 
        constraints: list[Constraint]
    ) -> dict[str, RetrievalResult]:
        """
        Retrieve evidence for multiple constraints.
        
        Args:
            constraints: List of constraints
            
        Returns:
            Dictionary mapping constraint_id to RetrievalResult
        """
        results = {}
        for constraint in constraints:
            results[constraint.id] = self.retrieve(constraint)
        return results
