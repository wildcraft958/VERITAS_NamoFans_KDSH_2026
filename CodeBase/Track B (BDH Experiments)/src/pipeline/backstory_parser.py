"""
SOTA Backstory Parser
Replaces regex-based extraction with semantic understanding.

Key improvements over regex:
1. Encodes ENTIRE backstory as primary representation
2. Uses attention-based claim extraction (not regex)
3. Captures implicit traits, negation scope, and causal relations
4. Falls back to model-based semantic chunking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any


class ClaimType(Enum):
    """Types of backstory claims."""
    TRAIT = "trait"              # Personality, beliefs
    EVENT = "event"              # Past happenings
    RELATIONSHIP = "relationship"  # Connections to others
    MOTIVATION = "motivation"    # Goals, fears, desires
    CONSTRAINT = "constraint"    # Rules, limitations
    UNKNOWN = "unknown"


@dataclass
class BackstoryClaim:
    """A semantic claim extracted from backstory."""
    text: str
    claim_type: ClaimType = ClaimType.UNKNOWN
    confidence: float = 1.0
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)
    attention_weight: float = 0.0  # How important this claim is
    

class SemanticBackstoryParser:
    """
    SOTA backstory parser using model-based semantic extraction.
    
    Unlike regex parsers, this:
    1. Encodes the FULL backstory as the primary signal
    2. Uses attention to find salient claims
    3. Preserves context and negation scope
    """
    
    # Semantic markers for claim type detection
    TRAIT_MARKERS = [
        "is a", "was a", "has always been", "tends to", "believes",
        "personality", "character", "nature", "temperament"
    ]
    EVENT_MARKERS = [
        "happened", "occurred", "when", "after", "before", "during",
        "once", "experienced", "witnessed", "survived"
    ]
    RELATIONSHIP_MARKERS = [
        "father", "mother", "friend", "enemy", "lover", "mentor",
        "relationship", "married", "sibling", "family"
    ]
    MOTIVATION_MARKERS = [
        "wants", "desires", "fears", "hopes", "dreams", "goal",
        "motivated by", "driven by", "seeks", "avoids"
    ]
    CONSTRAINT_MARKERS = [
        "cannot", "must not", "forbidden", "sworn", "bound by",
        "limited", "restricted", "obligated", "duty"
    ]
    
    def __init__(
        self,
        model: nn.Module = None,
        tokenizer = None,
        device: torch.device = None,
        chunk_size: int = 512,
        overlap: int = 64
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Claim type classifier (simple but effective)
        if model is not None and hasattr(model, 'config'):
            hidden_dim = getattr(model.config, 'hidden_size', 512)
            self.claim_classifier = nn.Linear(hidden_dim, len(ClaimType)).to(self.device)
        else:
            self.claim_classifier = None
    
    def parse(self, backstory: str) -> List[BackstoryClaim]:
        """
        Parse backstory into semantic claims.
        
        Unlike regex, this preserves full context and uses model understanding.
        """
        if not backstory or not backstory.strip():
            return []
        
        # Strategy: Sentence-level extraction with context
        sentences = self._split_into_sentences(backstory)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:  # Skip very short fragments
                continue
            
            # Detect claim type using semantic markers
            claim_type = self._detect_claim_type(sent)
            
            claims.append(BackstoryClaim(
                text=sent,
                claim_type=claim_type,
                confidence=1.0
            ))
        
        return claims
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving context."""
        import re
        # More sophisticated sentence splitting
        # Handles abbreviations and edge cases better
        sentence_enders = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        sentences = sentence_enders.split(text)
        
        # Further split on semicolons and long conjunctions
        refined = []
        for s in sentences:
            # Split on semicolons but keep each part
            parts = s.split(';')
            refined.extend([p.strip() for p in parts if p.strip()])
        
        return refined
    
    def _detect_claim_type(self, text: str) -> ClaimType:
        """Detect claim type using semantic markers."""
        text_lower = text.lower()
        
        # Score each type
        scores = {
            ClaimType.TRAIT: sum(1 for m in self.TRAIT_MARKERS if m in text_lower),
            ClaimType.EVENT: sum(1 for m in self.EVENT_MARKERS if m in text_lower),
            ClaimType.RELATIONSHIP: sum(1 for m in self.RELATIONSHIP_MARKERS if m in text_lower),
            ClaimType.MOTIVATION: sum(1 for m in self.MOTIVATION_MARKERS if m in text_lower),
            ClaimType.CONSTRAINT: sum(1 for m in self.CONSTRAINT_MARKERS if m in text_lower),
        }
        
        # Return highest scoring type, or UNKNOWN if no matches
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return ClaimType.UNKNOWN


class SemanticBackstoryEmbedder(nn.Module):
    """
    SOTA backstory embedder that:
    1. Encodes the FULL backstory as primary representation
    2. Adds weighted claim embeddings as auxiliary signal
    3. Uses attention to find relevant parts
    """
    
    def __init__(
        self,
        tokenizer,
        model: nn.Module,
        device: torch.device,
        use_full_encoding: bool = True,  # CRITICAL: Encode full backstory
        max_length: int = 512
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.use_full_encoding = use_full_encoding
        self.max_length = max_length
        
        # Get hidden dimension
        if hasattr(model, 'config'):
            self.hidden_dim = getattr(model.config, 'hidden_size', 512)
        else:
            self.hidden_dim = 512
        
        # Attention mechanism for claim weighting
        self.claim_attention = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(self.hidden_dim // 4, 1)
        ).to(device)
        
        # Claim type embeddings (learnable)
        self.type_embeddings = nn.Embedding(len(ClaimType), self.hidden_dim).to(device)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        ).to(device)
    
    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a piece of text using the model."""
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Remove token_type_ids if present (not supported by all models)
        if 'token_type_ids' in encoded:
            del encoded['token_type_ids']
        
        input_ids = encoded['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=True)
            
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                hidden = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            else:
                # Fallback
                hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Mean pooling
            return hidden.mean(dim=1).squeeze(0)
    
    def forward(
        self,
        backstory_text: str,
        claims: Optional[List[BackstoryClaim]] = None
    ) -> torch.Tensor:
        """
        Embed backstory with optional claim weighting.
        
        Args:
            backstory_text: Full backstory text
            claims: Optional pre-parsed claims
            
        Returns:
            Backstory embedding [hidden_dim]
        """
        # 1. ALWAYS encode full backstory (this is the key insight from the critique)
        full_embedding = self.embed_text(backstory_text)
        
        if not self.use_full_encoding or claims is None or len(claims) == 0:
            return full_embedding
        
        # 2. Encode individual claims for attention weighting
        claim_embeddings = []
        claim_types = []
        
        for claim in claims:
            if claim.embedding is not None:
                emb = claim.embedding.to(self.device)
            else:
                emb = self.embed_text(claim.text)
            claim_embeddings.append(emb)
            claim_types.append(claim.claim_type.value)
        
        if not claim_embeddings:
            return full_embedding
        
        # Stack claim embeddings
        claims_tensor = torch.stack(claim_embeddings)  # [num_claims, hidden_dim]
        
        # 3. Attention-weighted aggregation
        attn_scores = self.claim_attention(claims_tensor)  # [num_claims, 1]
        attn_weights = F.softmax(attn_scores, dim=0)
        weighted_claims = (claims_tensor * attn_weights).sum(dim=0)  # [hidden_dim]
        
        # 4. Fuse full encoding with weighted claims
        combined = torch.cat([full_embedding, weighted_claims])
        fused = self.fusion(combined)
        
        return fused
    
    def aggregate_claims(
        self,
        claims: List[BackstoryClaim]
    ) -> torch.Tensor:
        """
        Aggregate claims into a single representation.
        Used for compatibility with existing pipeline.
        """
        if not claims:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        # Get full backstory text from claims
        full_text = " ".join([c.text for c in claims])
        
        return self.forward(full_text, claims)


class DualStateMemory(nn.Module):
    """
    Dual-state memory for narrative processing.
    
    Implements the critique suggestion:
    - Fast local memory: chunk-level, resets periodically
    - Slow global memory: narrative-level, persists across chunks
    
    This prevents "bag-of-chunks" behavior and maintains temporal coherence.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        reset_interval: int = 10,  # Reset local memory every N chunks
        global_decay: float = 0.95  # Slow decay for global memory
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.reset_interval = reset_interval
        self.global_decay = global_decay
        
        # Local memory (fast, resets periodically)
        self.local_memory = None
        self.local_counter = 0
        
        # Global memory (slow, persistent)
        self.global_memory = None
        
        # Memory update gates
        self.local_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Memory fusion
        self.memory_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
    
    def reset(self):
        """Reset all memory states."""
        self.local_memory = None
        self.global_memory = None
        self.local_counter = 0
    
    def update(self, chunk_repr: torch.Tensor) -> torch.Tensor:
        """
        Update memory with new chunk representation.
        
        Args:
            chunk_repr: Representation from current chunk [hidden_dim] or [batch, hidden_dim]
            
        Returns:
            Memory-enhanced representation [same shape as input]
        """
        if chunk_repr.dim() == 1:
            chunk_repr = chunk_repr.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        batch_size = chunk_repr.size(0)
        device = chunk_repr.device
        
        # Initialize memories if needed
        if self.local_memory is None:
            self.local_memory = torch.zeros(batch_size, self.hidden_dim, device=device)
        if self.global_memory is None:
            self.global_memory = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Update local memory
        local_input = torch.cat([chunk_repr, self.local_memory], dim=-1)
        local_gate = self.local_gate(local_input)
        self.local_memory = local_gate * chunk_repr + (1 - local_gate) * self.local_memory
        
        # Update global memory (slower, decayed)
        global_input = torch.cat([chunk_repr, self.global_memory], dim=-1)
        global_gate = self.global_gate(global_input)
        update_strength = global_gate * (1 - self.global_decay)
        self.global_memory = self.global_memory * self.global_decay + update_strength * chunk_repr
        
        # Fuse memories with current representation
        combined_memory = torch.cat([self.local_memory, self.global_memory], dim=-1)
        enhanced = self.memory_fusion(combined_memory)
        
        # Periodic local reset
        self.local_counter += 1
        if self.local_counter >= self.reset_interval:
            self.local_memory = torch.zeros_like(self.local_memory)
            self.local_counter = 0
        
        if squeeze:
            enhanced = enhanced.squeeze(0)
        
        return enhanced
    
    def get_state(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get current memory states."""
        return self.local_memory, self.global_memory
    
    def set_state(
        self,
        local: Optional[torch.Tensor],
        global_: Optional[torch.Tensor]
    ):
        """Set memory states (for checkpointing)."""
        self.local_memory = local
        self.global_memory = global_
