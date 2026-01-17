"""
Chunk Aggregation Module
Aggregates variable-length chunk representations into fixed-size narrative representations.
Supports multiple aggregation strategies based on the critique:
- Mean Pooling (baseline, washes out signals)
- Max Pooling (detects outliers/contradictions)
- Attention Pooling (learns which chunks matter)
- Top-K Selection (keeps most relevant chunks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Literal


class ChunkAggregator(nn.Module):
    """
    Aggregates a variable number of chunk representations into a single representation.
    
    This is the critical component that prevents "single-vector collapse" by using
    more sophisticated aggregation than naive mean pooling.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        method: Literal["mean", "max", "attention", "topk"] = "max",
        num_heads: int = 4,
        dropout: float = 0.1,
        top_k: int = 5
    ):
        """
        Args:
            hidden_dim: Dimension of chunk representations
            method: Aggregation method - "mean", "max", "attention", or "topk"
            num_heads: Number of attention heads (for attention method)
            dropout: Dropout rate
            top_k: Number of top chunks to keep (for topk method)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method
        self.top_k = top_k
        
        if method == "attention":
            # Learnable query vector for attention pooling
            self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.layer_norm = nn.LayerNorm(hidden_dim)
        
        elif method == "topk":
            # Scoring network to select top-k chunks
            self.scorer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
    
    def forward(
        self,
        chunk_reprs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate chunk representations.
        
        Args:
            chunk_reprs: Tensor of shape [num_chunks, hidden_dim] or [batch, num_chunks, hidden_dim]
            mask: Optional mask of shape [num_chunks] or [batch, num_chunks], 1 for valid, 0 for padding
            
        Returns:
            Aggregated representation of shape [hidden_dim] or [batch, hidden_dim]
        """
        # Handle 2D input (single narrative)
        if chunk_reprs.dim() == 2:
            chunk_reprs = chunk_reprs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, num_chunks, hidden_dim = chunk_reprs.shape
        
        if self.method == "mean":
            if mask is not None:
                # Masked mean
                mask = mask.unsqueeze(-1).float()  # [batch, num_chunks, 1]
                output = (chunk_reprs * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                output = chunk_reprs.mean(dim=1)
        
        elif self.method == "max":
            if mask is not None:
                # Replace masked positions with -inf before max
                mask = mask.unsqueeze(-1).bool()  # [batch, num_chunks, 1]
                chunk_reprs = chunk_reprs.masked_fill(~mask, float('-inf'))
            output, _ = chunk_reprs.max(dim=1)
            # Handle case where all are masked (replace -inf with 0)
            output = torch.where(torch.isinf(output), torch.zeros_like(output), output)
        
        elif self.method == "attention":
            # Expand query for batch
            query = self.query.expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
            
            # Create attention mask if provided
            attn_mask = None
            if mask is not None:
                # For MultiheadAttention, key_padding_mask should be True for positions to mask
                attn_mask = ~mask.bool()  # [batch, num_chunks]
            
            # Attention pooling: query attends to chunks
            attended, _ = self.attention(
                query=query,
                key=chunk_reprs,
                value=chunk_reprs,
                key_padding_mask=attn_mask
            )  # [batch, 1, hidden_dim]
            
            output = self.layer_norm(attended.squeeze(1))  # [batch, hidden_dim]
        
        elif self.method == "topk":
            # Score each chunk
            scores = self.scorer(chunk_reprs).squeeze(-1)  # [batch, num_chunks]
            
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), float('-inf'))
            
            # Get top-k indices
            k = min(self.top_k, num_chunks)
            _, top_indices = torch.topk(scores, k, dim=1)  # [batch, k]
            
            # Gather top-k chunk representations
            top_indices = top_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            top_chunks = torch.gather(chunk_reprs, 1, top_indices)  # [batch, k, hidden_dim]
            
            # Max pool over top-k
            output, _ = top_chunks.max(dim=1)  # [batch, hidden_dim]
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")
        
        if squeeze_output:
            output = output.squeeze(0)
        
        return output


class HierarchicalAggregator(nn.Module):
    """
    Two-level aggregation: chunks -> sections -> narrative.
    Useful for very long narratives with natural structure.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        chunk_to_section_method: str = "max",
        section_to_narrative_method: str = "attention",
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.chunk_aggregator = ChunkAggregator(
            hidden_dim=hidden_dim,
            method=chunk_to_section_method,
            dropout=dropout
        )
        
        self.section_aggregator = ChunkAggregator(
            hidden_dim=hidden_dim,
            method=section_to_narrative_method,
            num_heads=num_heads,
            dropout=dropout
        )
    
    def forward(
        self,
        chunk_reprs: torch.Tensor,
        section_boundaries: List[int]
    ) -> torch.Tensor:
        """
        Hierarchical aggregation.
        
        Args:
            chunk_reprs: [num_chunks, hidden_dim]
            section_boundaries: List of indices where sections end
            
        Returns:
            Aggregated representation [hidden_dim]
        """
        # Split chunks into sections
        sections = []
        start = 0
        for end in section_boundaries:
            if start < end:
                section_chunks = chunk_reprs[start:end]
                section_repr = self.chunk_aggregator(section_chunks)
                sections.append(section_repr)
            start = end
        
        # Handle remaining chunks
        if start < len(chunk_reprs):
            section_chunks = chunk_reprs[start:]
            section_repr = self.chunk_aggregator(section_chunks)
            sections.append(section_repr)
        
        if not sections:
            return torch.zeros(chunk_reprs.shape[-1], device=chunk_reprs.device)
        
        # Aggregate sections
        section_tensor = torch.stack(sections)  # [num_sections, hidden_dim]
        return self.section_aggregator(section_tensor)
