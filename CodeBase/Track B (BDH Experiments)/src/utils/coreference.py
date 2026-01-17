"""
Global Coreference Resolution Module
===================================

Identifies and resolves coreferences (pronouns -> nouns) in the text
BEFORE it hits the BDH pipeline.

Why?
"He went to the cave." -> Ambiguous.
"Thalcave went to the cave." -> Checkable.

Uses: fastcoref (F-Coreference)
"""

import logging
from typing import List, Optional
import torch

try:
    from fastcoref import FCoref
    _FASTCOREF_AVAILABLE = True
except ImportError:
    _FASTCOREF_AVAILABLE = False


class GlobalCoreferenceResolver:
    """
    Wrapper for FastCoref to resolve entities in narratives.
    """
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        if _FASTCOREF_AVAILABLE:
            try:
                # Load the lighter model by default for speed
                self.model = FCoref(device=self.device)
                print(f"GlobalCoreferenceResolver: Loaded FCoref on {self.device}")
            except Exception as e:
                print(f"GlobalCoreferenceResolver: Failed to load FCoref: {e}")
                # Do not modify global state here
                self.model = None
        else:
            print("GlobalCoreferenceResolver: fastcoref not installed. Using fallback (identity).")
            
    def resolve(self, text: str) -> str:
        """
        Resolve coreferences in the text.
        Replaces pronouns with their referents.
        """
        if not _FASTCOREF_AVAILABLE or self.model is None:
            return text
            
        try:
            # fastcoref returns a list of resolved texts (one per input)
            preds = self.model.predict(texts=[text])
            return preds[0].get_resolved_text()
        except Exception as e:
            logging.warning(f"Coreference resolution failed: {e}")
            return text

    def batch_resolve(self, texts: List[str]) -> List[str]:
        """
        Batch resolve for higher throughput.
        """
        if not _FASTCOREF_AVAILABLE or self.model is None:
            return texts
            
        try:
            preds = self.model.predict(texts=texts)
            return [p.get_resolved_text() for p in preds]
        except Exception as e:
            logging.warning(f"Batch coreference resolution failed: {e}")
            return texts
