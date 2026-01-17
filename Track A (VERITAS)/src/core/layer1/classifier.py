"""
Zero-shot constraint classification.

Classifies atomic facts into constraint types using BART-MNLI.
Based on: https://huggingface.co/facebook/bart-large-mnli
"""

from functools import lru_cache

from core.layer1.constraint_graph import AtomicFact, ConstraintType


# Lazy loading for transformers to avoid slow imports
_classifier = None


def _get_classifier():
    """Lazy load the HuggingFace classifier."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=-1  # CPU by default, change to 0 for GPU
        )
    return _classifier


class ConstraintClassifier:
    """
    Zero-shot classifier for constraint types.
    
    Uses facebook/bart-large-mnli pre-trained on MNLI, FEVER, and ANLI
    (~3M examples) for zero-shot classification without any training.
    
    Repository: https://huggingface.co/facebook/bart-large-mnli
    """
    
    # Label descriptions for better classification
    LABEL_TEMPLATES = {
        ConstraintType.TEMPORAL: "This is about time, age, sequence, or when events happened",
        ConstraintType.PSYCHOLOGICAL: "This is about personality, skills, beliefs, traits, or character",
        ConstraintType.CAUSAL: "This is about events, actions, consequences, or what happened",
        ConstraintType.WORLD_MODEL: "This is about how the world works, rules, or assumptions"
    }
    
    LABELS = list(LABEL_TEMPLATES.keys())
    LABEL_NAMES = [ct.value for ct in LABELS]
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the classifier.
        
        Args:
            use_gpu: Whether to use GPU for inference
        """
        self.use_gpu = use_gpu
    
    def classify(self, fact: AtomicFact) -> tuple[ConstraintType, float]:
        """
        Classify an atomic fact into a constraint type.
        
        Args:
            fact: The atomic fact to classify
            
        Returns:
            Tuple of (ConstraintType, confidence_score)
        """
        classifier = _get_classifier()
        
        result = classifier(
            fact.text,
            candidate_labels=self.LABEL_NAMES,
            hypothesis_template="{}."
        )
        
        # Get top label and score
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        
        # Map back to ConstraintType
        constraint_type = ConstraintType(top_label)
        
        return constraint_type, top_score
    
    def classify_batch(
        self, 
        facts: list[AtomicFact]
    ) -> list[tuple[ConstraintType, float]]:
        """
        Classify multiple facts at once (more efficient).
        
        Args:
            facts: List of atomic facts
            
        Returns:
            List of (ConstraintType, confidence) tuples
        """
        classifier = _get_classifier()
        texts = [f.text for f in facts]
        
        results = classifier(
            texts,
            candidate_labels=self.LABEL_NAMES,
            hypothesis_template="{}."
        )
        
        # Handle single result (transformers returns dict instead of list for single item)
        if isinstance(results, dict):
            results = [results]
        
        classifications = []
        for result in results:
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            constraint_type = ConstraintType(top_label)
            classifications.append((constraint_type, top_score))
        
        return classifications
    
    def get_all_scores(self, fact: AtomicFact) -> dict[ConstraintType, float]:
        """
        Get classification scores for all constraint types.
        
        Args:
            fact: The atomic fact to classify
            
        Returns:
            Dictionary mapping ConstraintType to confidence score
        """
        classifier = _get_classifier()
        
        result = classifier(
            fact.text,
            candidate_labels=self.LABEL_NAMES,
            hypothesis_template="{}."
        )
        
        scores = {}
        for label, score in zip(result["labels"], result["scores"]):
            constraint_type = ConstraintType(label)
            scores[constraint_type] = score
        
        return scores
