"""
BDH Scanner - SOTA BDH-GPU Implementation
==========================================

Implements the full "BDH-GPU" architecture from the paper (2509.26507v1):
1. Online Hebbian Update: State += Lambda * State + Outer(Key, Value)
2. Sparse Positive Activations: ReLU + TopK thresholding
3. Inhibitory Circuits: Excitatory + Inhibitory updates (Section 2.5)
4. Symbolic Injection: Causal graph constraints embedded in matrix (Section 5.4)
5. Synaptic Drift: Drift = ||Value - Prediction||

Key Upgrades from v1.0:
- Inhibitory update rule for contradiction detection
- inject_causal_graph() for neuro-symbolic reasoning
- sparsify() helper function exportable for pipeline use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class ConsistencyScore:
    """Result of consistency check."""
    consistent: bool
    perplexity: float  # kept for compat
    surprise_score: float  # Normalized [0, 1] where 1 = maximum contradiction
    confidence: float
    evidence_chunks: List[Dict[str, Any]]  # Which chunks contributed to decision


def sparsify(vector: torch.Tensor, k: float = 0.05) -> torch.Tensor:
    """
    Enforce Sparse Positive Activations (ReLU + TopK).
    Paper Section 6.4: "Activation vectors of BDH are sparse and positive."
    
    Args:
        vector: Input tensor [Dim] or [Batch, Dim]
        k: Fraction of neurons to keep active (default: 5%)
        
    Returns:
        Sparsified tensor with only top k% neurons active
    """
    # 1. Positive (ReLU)
    positive = F.relu(vector)
    
    if positive.dim() == 1:
        # Single vector mode
        dim = positive.size(0)
        top_k = max(1, int(dim * k))
        threshold = torch.kthvalue(positive, dim - top_k + 1).values
        mask = (positive >= threshold).float()
        return positive * mask
    else:
        # Batch mode [Batch, Dim]
        dim = positive.size(-1)
        top_k = max(1, int(dim * k))
        thresholds = torch.kthvalue(positive, dim - top_k + 1, dim=-1).values
        mask = (positive >= thresholds.unsqueeze(-1)).float()
        return positive * mask


class BDHScanner(nn.Module):
    """
    SOTA BDH-GPU Scanner with Neuro-Symbolic Integration.
    
    Implements biologically plausible graph dynamics with:
    - Hebbian learning (excitatory)
    - Inhibitory circuits (contradiction suppression)
    - Symbolic constraint injection (causal graph)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        decay: float = 0.99,  # Lambda forgetting factor
        sparsity: float = 0.05,  # Top 5% neurons active
        inhibition_factor: float = 1.5,  # Strength of inhibitory feedback
        device: torch.device = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay = decay
        self.sparsity = sparsity
        self.inhibition_factor = inhibition_factor
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # The "Synaptic State" (Long-term memory within the context window)
        # In BDH paper, this is matrix sigma (n x n)
        self.state = torch.zeros(
            (hidden_dim, hidden_dim), 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Vocabulary mapping for symbolic injection
        self._vocab_to_idx: Dict[str, int] = {}
        self._next_vocab_idx = 0
        
    def reset_state(self):
        """Clear short-term synaptic memory."""
        self.state.zero_()
        
    def _sparse_activation(self, vector: torch.Tensor) -> torch.Tensor:
        """Apply sparse positive activation using module-level sparsify."""
        return sparsify(vector.to(self.device), k=self.sparsity)

    def step(
        self, 
        key_vector: torch.Tensor, 
        value_vector: torch.Tensor,
        contradiction_score: float = 0.0
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform one Hebbian update step with inhibitory feedback (Paper Section 2.5, 3.2).
        
        Args:
            key_vector: Stimulus [Dim]
            value_vector: Response [Dim]
            contradiction_score: NLI contradiction probability [0, 1]
                                If > 0, activates inhibitory circuit
            
        Returns:
            drift_score: Euclidean distance between Prediction and Reality
            current_state_snapshot: Clone of the synaptic matrix (for viz)
        """
        # Ensure inputs are on device and sparse/positive
        k = self._sparse_activation(key_vector.to(self.device))
        v = self._sparse_activation(value_vector.to(self.device))
        
        # 1. Prediction: What did the brain expect?
        # Expectation = Key * State (Standard Linear Attention retrieval)
        expectation = torch.matmul(k, self.state)
        
        # 2. Calculate Surprise (Synaptic Drift)
        error_signal = v - expectation
        drift_score = torch.norm(error_signal, p=2).item()
        
        # 3. Hebbian Learning Update with Inhibitory Circuit
        # Paper Section 2.5: "Expressing BDH using brain models"
        
        # 3a. Excitatory Update (Standard Hebbian: "Fire together, wire together")
        excitatory_signal = torch.outer(k, v)
        
        # 3b. Inhibitory Update (Contradiction suppression)
        # If NLI says "Contradiction", we actively depress the synapse
        inhibitory_signal = excitatory_signal * contradiction_score * self.inhibition_factor
        
        # 3c. Combined Update
        update = excitatory_signal - inhibitory_signal
        
        # 4. Apply with Decay
        self.state = (self.state * self.decay) + update
        
        # 5. Enforce Non-Negativity (ReLU constraint from BDH-GPU)
        self.state = F.relu(self.state)
        
        return drift_score, self.state.clone().cpu()

    def step_with_nli(
        self, 
        key_vector: torch.Tensor, 
        value_vector: torch.Tensor,
        nli_scores: Optional[Dict[str, float]] = None
    ) -> Tuple[float, torch.Tensor]:
        """
        Convenience wrapper that extracts contradiction score from NLI result.
        
        Args:
            key_vector: Stimulus [Dim]
            value_vector: Response [Dim]
            nli_scores: Dict with 'contradiction', 'entailment', 'neutral' probabilities
            
        Returns:
            drift_score, state_snapshot
        """
        contradiction_score = 0.0
        if nli_scores is not None:
            contradiction_score = nli_scores.get('contradiction', 0.0)
        
        return self.step(key_vector, value_vector, contradiction_score)

    def inject_causal_graph(
        self, 
        causal_events: List[Tuple[str, str]], 
        injection_strength: float = 10.0
    ):
        """
        Hard-code symbolic logical constraints into the neural matrix.
        Paper Section 5.4: "Modularity in BDH-GPU signal propagation"
        
        This creates "monosemantic synapses" that represent causal relationships.
        If 'A causes B', we artificially spike the weight W[A, B].
        
        Args:
            causal_events: List of (event_A, event_B) tuples where A -> B
            injection_strength: How strongly to encode the relationship
        """
        for event_a, event_b in causal_events:
            idx_a = self._get_or_create_idx(event_a)
            idx_b = self._get_or_create_idx(event_b)
            
            # Only inject if both indices fit in our hidden_dim
            if idx_a < self.hidden_dim and idx_b < self.hidden_dim:
                # Manually strengthen the synaptic connection
                # This creates a "bias" that the neural model must respect
                self.state[idx_a, idx_b] += injection_strength
    
    def inject_contradiction(
        self, 
        event_a: str, 
        event_b: str, 
        suppression_strength: float = -10.0
    ):
        """
        Inject a known contradiction into the matrix (inhibitory).
        
        If 'A contradicts B', we artificially suppress the weight W[A, B].
        
        Args:
            event_a: First event
            event_b: Second event (contradicts A)
            suppression_strength: Negative value to suppress connection
        """
        idx_a = self._get_or_create_idx(event_a)
        idx_b = self._get_or_create_idx(event_b)
        
        if idx_a < self.hidden_dim and idx_b < self.hidden_dim:
            self.state[idx_a, idx_b] += suppression_strength
            # Ensure non-negativity after suppression
            self.state = F.relu(self.state)
    
    def _get_or_create_idx(self, event: str) -> int:
        """Get vocabulary index for an event, creating if needed."""
        if event not in self._vocab_to_idx:
            self._vocab_to_idx[event] = self._next_vocab_idx
            self._next_vocab_idx = (self._next_vocab_idx + 1) % self.hidden_dim
        return self._vocab_to_idx[event]

    def process_sequence(
        self, 
        embeddings: torch.Tensor,
        contradiction_scores: Optional[List[float]] = None
    ) -> Tuple[List[float], List[torch.Tensor]]:
        """
        Process a sequence of embeddings token-by-token (Online Learning).
        
        Args:
            embeddings: [Seq_Len, Hidden_Dim] tensor
            contradiction_scores: Optional per-token contradiction probabilities
            
        Returns:
            drift_scores: List of drift values per token
            states: List of state snapshots
        """
        drift_scores = []
        states = []
        
        seq_len = embeddings.size(0)
        
        # Default to no contradictions if not provided
        if contradiction_scores is None:
            contradiction_scores = [0.0] * seq_len
        
        # Pad or truncate contradiction scores
        if len(contradiction_scores) < seq_len:
            contradiction_scores = contradiction_scores + [0.0] * (seq_len - len(contradiction_scores))
        
        # For auto-associative memory, Key = Value = Input Embedding
        for i in range(seq_len):
            vector = embeddings[i]
            contra_score = contradiction_scores[i] if i < len(contradiction_scores) else 0.0
            drift, state_snapshot = self.step(vector, vector, contra_score)
            drift_scores.append(drift)
            states.append(state_snapshot)
            
        return drift_scores, states

    def get_state_statistics(self) -> Dict[str, float]:
        """Get statistics about the current synaptic state."""
        state = self.state
        return {
            'mean': state.mean().item(),
            'std': state.std().item(),
            'max': state.max().item(),
            'min': state.min().item(),
            'sparsity': (state == 0).float().mean().item(),
            'nonzero_fraction': (state > 0).float().mean().item(),
        }
