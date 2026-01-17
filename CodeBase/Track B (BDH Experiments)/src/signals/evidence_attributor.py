"""
Evidence Attribution Module

Implements the blueprint's requirement for:
1. Tracking WHICH passage contradicts WHICH claim
2. Cross-attention between backstory claims and narrative chunks
3. Producing interpretable evidence for the verdict

This is critical for the "Evidence Rationale" output required by Track B.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import math


@dataclass
class EvidenceChunk:
    """A chunk of narrative text with its relevance to a claim."""
    text: str
    chunk_index: int
    start_position: int
    end_position: int
    relevance_score: float
    attention_weight: float
    is_contradicting: bool = False
    is_supporting: bool = False


@dataclass
class ClaimEvidence:
    """Evidence for/against a specific backstory claim."""
    claim_text: str
    verdict: str  # "supports", "contradicts", "neutral"
    confidence: float
    supporting_chunks: List[EvidenceChunk]
    contradicting_chunks: List[EvidenceChunk]
    explanation: str


class CrossAttentionClassifier(nn.Module):
    """
    Cross-attention based classifier that directly attends backstory claims
    to narrative chunks.
    
    This enables:
    1. Fine-grained evidence attribution
    2. Interpretable attention maps
    3. Better signal than mean/max pooling
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project narrative and backstory to same space
        self.narrative_proj = nn.Linear(hidden_dim, hidden_dim)
        self.backstory_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Cross-attention: backstory (query) attends to narrative (key, value)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Self-attention for refining representations
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feedforward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Consistent / Contradict
        )
        
        # Attention weights storage for interpretability
        self.last_attention_weights = None
    
    def forward(
        self,
        narrative_chunks: torch.Tensor,  # [batch, num_chunks, hidden_dim]
        backstory_claims: torch.Tensor,  # [batch, num_claims, hidden_dim]
        narrative_mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with cross-attention for evidence attribution.
        
        Args:
            narrative_chunks: Narrative chunk embeddings
            backstory_claims: Backstory claim embeddings
            narrative_mask: Mask for padding in narrative chunks
            return_attention: If True, return attention weights for interpretability
            
        Returns:
            (logits, probs, attention_weights)
        """
        batch_size = narrative_chunks.size(0)
        
        # Project to common space
        narrative = self.narrative_proj(narrative_chunks)  # [B, N_chunks, D]
        backstory = self.backstory_proj(backstory_claims)  # [B, N_claims, D]
        
        # Cross-attention: backstory attends to narrative
        # This is the key: each backstory claim "looks for" relevant narrative evidence
        attended_narrative, attn_weights = self.cross_attention(
            query=backstory,
            key=narrative,
            value=narrative,
            key_padding_mask=narrative_mask,
            need_weights=True,
            average_attn_weights=False  # Keep per-head weights for analysis
        )
        # attn_weights: [B, num_heads, num_claims, num_chunks]
        
        # Store for interpretability
        if return_attention:
            self.last_attention_weights = attn_weights.detach()
        
        # Residual + LayerNorm
        backstory = self.ln1(backstory + attended_narrative)
        
        # Self-attention on backstory (claims can attend to each other)
        backstory_refined, _ = self.self_attention(backstory, backstory, backstory)
        backstory = self.ln2(backstory + backstory_refined)
        
        # Feedforward
        backstory = self.ln3(backstory + self.ffn(backstory))
        
        # Aggregate: pool over claims
        backstory_pooled = backstory.mean(dim=1)  # [B, D]
        narrative_pooled = narrative.mean(dim=1)   # [B, D]
        
        # Concatenate for classification
        combined = torch.cat([backstory_pooled, narrative_pooled], dim=-1)
        
        logits = self.classifier(combined)
        probs = F.softmax(logits, dim=-1)
        
        if return_attention:
            # Average over heads for interpretability
            avg_attn = attn_weights.mean(dim=1)  # [B, num_claims, num_chunks]
            return logits, probs, avg_attn
        
        return logits, probs, None


class EvidenceRetriever:
    """
    Retrieves and ranks evidence from narrative for backstory claims.
    
    Uses both:
    1. Embedding similarity (semantic relevance)
    2. Attention weights from classifier (learned relevance)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        classifier: Optional[CrossAttentionClassifier] = None,
        device: torch.device = None,
        chunk_size: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.classifier = classifier
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        
        # Storage
        self.narrative_chunks: List[str] = []
        self.narrative_embeddings: Optional[torch.Tensor] = None
    
    def index_narrative(self, narrative_text: str):
        """
        Index narrative into chunks with embeddings.
        
        Args:
            narrative_text: Full novel text
        """
        # Chunk the narrative
        self.narrative_chunks = self._chunk_text(narrative_text)
        
        # Embed each chunk
        embeddings = []
        self.model.eval()
        
        for chunk in self.narrative_chunks:
            tokens = self.tokenizer(
                chunk,
                return_tensors='pt',
                max_length=self.chunk_size,
                truncation=True,
                padding=True
            )['input_ids'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=tokens, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                emb = hidden.mean(dim=1).squeeze(0)
                embeddings.append(emb.cpu())
        
        self.narrative_embeddings = torch.stack(embeddings)  # [N_chunks, D]
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments."""
        words = text.split()
        chunks = []
        
        stride = self.chunk_size // 2  # 50% overlap
        for i in range(0, len(words), stride):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) > 50:  # Skip very short chunks
                chunks.append(' '.join(chunk_words))
        
        return chunks
    
    def retrieve_evidence(
        self,
        backstory_text: str,
        claims: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[ClaimEvidence]:
        """
        Retrieve evidence for backstory from indexed narrative.
        
        Args:
            backstory_text: Full backstory text
            claims: Optional list of individual claims (if already parsed)
            top_k: Number of top chunks to return per claim
            
        Returns:
            List of ClaimEvidence objects
        """
        if self.narrative_embeddings is None:
            raise ValueError("Narrative not indexed. Call index_narrative first.")
        
        # Parse claims if not provided
        if claims is None:
            claims = self._parse_claims(backstory_text)
        
        evidence_list = []
        
        for claim in claims:
            # Embed claim
            claim_emb = self._embed_text(claim)
            
            # Compute similarity to all chunks
            similarities = F.cosine_similarity(
                claim_emb.unsqueeze(0),
                self.narrative_embeddings.to(claim_emb.device)
            )
            
            # Get top-k chunks
            top_indices = torch.topk(similarities, min(top_k, len(self.narrative_chunks))).indices
            
            # Create evidence chunks
            supporting = []
            contradicting = []
            
            for idx in top_indices:
                idx = idx.item()
                chunk = EvidenceChunk(
                    text=self.narrative_chunks[idx][:500] + "...",  # Truncate for display
                    chunk_index=idx,
                    start_position=idx * self.chunk_size // 2,
                    end_position=(idx + 1) * self.chunk_size // 2,
                    relevance_score=similarities[idx].item(),
                    attention_weight=0.0,  # Will be filled by classifier
                    is_contradicting=False,
                    is_supporting=similarities[idx].item() > 0.5
                )
                
                if chunk.is_supporting:
                    supporting.append(chunk)
                else:
                    contradicting.append(chunk)
            
            # Create claim evidence
            evidence_list.append(ClaimEvidence(
                claim_text=claim,
                verdict="neutral",  # Will be refined by classifier
                confidence=0.5,
                supporting_chunks=supporting,
                contradicting_chunks=contradicting,
                explanation=""
            ))
        
        return evidence_list
    
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed a piece of text."""
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.chunk_size,
            truncation=True,
            padding=True
        )['input_ids'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            return hidden.mean(dim=1).squeeze(0)
    
    def _parse_claims(self, backstory_text: str) -> List[str]:
        """Parse backstory into individual claims."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', backstory_text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def get_attention_attribution(
        self,
        narrative_chunks: torch.Tensor,
        backstory_claims: torch.Tensor
    ) -> List[List[Tuple[int, float]]]:
        """
        Get attention-based attribution from classifier.
        
        Returns:
            For each claim, list of (chunk_idx, attention_weight) tuples
        """
        if self.classifier is None:
            return []
        
        self.classifier.eval()
        with torch.no_grad():
            _, _, attn = self.classifier(
                narrative_chunks.unsqueeze(0),
                backstory_claims.unsqueeze(0),
                return_attention=True
            )
        
        # attn: [1, num_claims, num_chunks]
        attn = attn.squeeze(0)  # [num_claims, num_chunks]
        
        attributions = []
        for claim_idx in range(attn.size(0)):
            claim_attn = attn[claim_idx]
            # Get top chunks by attention
            top_k = min(5, claim_attn.size(0))
            top_indices = torch.topk(claim_attn, top_k).indices
            
            claim_attribution = [
                (idx.item(), claim_attn[idx].item())
                for idx in top_indices
            ]
            attributions.append(claim_attribution)
        
        return attributions
