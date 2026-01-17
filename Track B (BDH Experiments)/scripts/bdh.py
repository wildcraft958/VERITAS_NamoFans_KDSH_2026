"""
BDH Narrative Consistency - Abstract Interface
==============================================

Provides a clean, packageable interface for the BDH pipeline.

Usage:
    from bdh import NarrativeConsistencyChecker
    
    checker = NarrativeConsistencyChecker()
    checker.load_novel("path/to/novel.txt")
    result = checker.check("The character was born in Paris...")
    
    print(result.consistent)    # True/False
    print(result.confidence)    # 0.0-1.0
    print(result.rationale)     # Explanation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class ConsistencyResult:
    """Result of consistency check."""
    consistent: bool
    confidence: float
    rationale: str
    scores: Dict[str, float]  # Individual signal scores
    evidence: List[str]  # Supporting/contradicting passages


class BaseNarrativeChecker(ABC):
    """Abstract base class for narrative consistency checkers."""
    
    @abstractmethod
    def load_novel(self, novel_path: str) -> None:
        """Load and process a novel."""
        pass
    
    @abstractmethod
    def load_novel_text(self, novel_text: str) -> None:
        """Load and process novel from text directly."""
        pass
    
    @abstractmethod
    def check(self, backstory: str) -> ConsistencyResult:
        """Check if backstory is consistent with loaded novel."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for new novel."""
        pass


class NarrativeConsistencyChecker(BaseNarrativeChecker):
    """
    Main interface for BDH narrative consistency checking.
    
    Example:
        checker = NarrativeConsistencyChecker(device="cuda")
        checker.load_novel("novel.txt")
        result = checker.check("Alice was born in London...")
        print(f"Consistent: {result.consistent}")
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the checker.
        
        Args:
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self._pipeline = None
        self._novel_loaded = False
    
    def _ensure_pipeline(self):
        """Lazy initialization of pipeline."""
        if self._pipeline is None:
            from src.pipeline import create_sota_pipeline
            self._pipeline = create_sota_pipeline(device=self.device)
    
    def load_novel(self, novel_path: str) -> None:
        """
        Load a novel from file.
        
        Args:
            novel_path: Path to .txt file
        """
        path = Path(novel_path)
        if not path.exists():
            raise FileNotFoundError(f"Novel not found: {novel_path}")
        
        text = path.read_text(encoding='utf-8', errors='replace')
        self.load_novel_text(text)
    
    def load_novel_text(self, novel_text: str) -> None:
        """
        Load a novel from text.
        
        Args:
            novel_text: Full novel text
        """
        self._ensure_pipeline()
        self._pipeline.ingest_novel(novel_text)
        self._novel_loaded = True
    
    def check(self, backstory: str) -> ConsistencyResult:
        """
        Check if backstory is consistent with loaded novel.
        
        Args:
            backstory: The hypothetical backstory to check
            
        Returns:
            ConsistencyResult with verdict and evidence
        """
        if not self._novel_loaded:
            raise RuntimeError("No novel loaded. Call load_novel() first.")
        
        self._ensure_pipeline()
        verdict = self._pipeline.check_consistency(backstory)
        
        return ConsistencyResult(
            consistent=verdict.consistent,
            confidence=verdict.confidence,
            rationale=getattr(verdict, 'rationale', ''),
            scores=verdict.scores if hasattr(verdict, 'scores') else {},
            evidence=[]  # TODO: Extract from verdict
        )
    
    def reset(self) -> None:
        """Reset for a new novel."""
        if self._pipeline:
            self._pipeline.novel_indexed = False
            self._pipeline.novel_text = ""
        self._novel_loaded = False
    
    def __repr__(self):
        status = "loaded" if self._novel_loaded else "no novel"
        return f"NarrativeConsistencyChecker(device={self.device}, status={status})"


# Convenience function
def check_consistency(
    novel_path: str, 
    backstory: str, 
    device: str = "cuda"
) -> ConsistencyResult:
    """
    One-shot consistency check.
    
    Args:
        novel_path: Path to novel .txt file
        backstory: Backstory text to check
        device: 'cuda' or 'cpu'
        
    Returns:
        ConsistencyResult
        
    Example:
        result = check_consistency("novel.txt", "Alice was born...")
        print(result.consistent)
    """
    checker = NarrativeConsistencyChecker(device=device)
    checker.load_novel(novel_path)
    return checker.check(backstory)


# For package-level import
__all__ = [
    'NarrativeConsistencyChecker',
    'ConsistencyResult',
    'BaseNarrativeChecker',
    'check_consistency',
]
