"""
Cross-Encoder Evidence Attribution

SOTA evidence attribution using cross-encoder models for precise
claim-to-passage matching and interpretability.

Key features:
- Cross-encoder (not bi-encoder) for accurate similarity
- Claim decomposition for granular matching
- Passage reranking for top evidence
- Interpretable evidence chains
- Direct claim-passage entailment/contradiction scoring
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import re


@dataclass
class EvidencePassage:
    """A passage of text that serves as evidence."""
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    
    # Scores for this passage
    relevance_score: float = 0.0
    entailment_score: float = 0.0
    contradiction_score: float = 0.0
    neutral_score: float = 0.0
    
    # Final verdict for this passage
    verdict: str = "neutral"  # entailment, contradiction, neutral


@dataclass
class ClaimEvidence:
    """Evidence for a single claim."""
    claim_text: str
    
    # Top evidence passages
    supporting_passages: List[EvidencePassage] = field(default_factory=list)
    contradicting_passages: List[EvidencePassage] = field(default_factory=list)
    neutral_passages: List[EvidencePassage] = field(default_factory=list)
    
    # Overall verdict for this claim
    verdict: str = "neutral"
    confidence: float = 0.0
    
    # Best evidence text for display
    best_supporting: Optional[str] = None
    best_contradicting: Optional[str] = None


@dataclass
class EvidenceReport:
    """Full evidence report for backstory."""
    claim_evidence: List[ClaimEvidence]
    
    # Summary statistics
    total_claims: int = 0
    supported_claims: int = 0
    contradicted_claims: int = 0
    neutral_claims: int = 0
    
    # Top evidence across all claims
    top_supporting_passages: List[Dict[str, Any]] = field(default_factory=list)
    top_contradicting_passages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Overall consistency
    is_consistent: bool = True
    confidence: float = 0.5


class CrossEncoderEvidenceAttributor:
    """
    SOTA Evidence Attribution using Cross-Encoder.
    
    Unlike bi-encoders that encode query and passage separately,
    cross-encoders encode them together for more accurate matching.
    
    Model: cross-encoder/nli-deberta-v3-base
    - Trained specifically for NLI-style claim-passage scoring
    - Outputs entailment/neutral/contradiction
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: torch.device = None,
        chunk_size: int = 300,  # Words per chunk
        chunk_overlap: int = 50,
        batch_size: int = 16,
        top_k_evidence: int = 5,
    ):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.top_k = top_k_evidence
        
        self.model = None
        self._loaded = False
        
        # Storage
        self.narrative_chunks: List[EvidencePassage] = []
    
    def load(self):
        """Load cross-encoder model."""
        if self._loaded:
            return
        
        print(f"Loading cross-encoder: {self.model_name}")
        
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=str(self.device)
            )
            print(f"Cross-encoder loaded on {self.device}")
            self._loaded = True
        except ImportError:
            print("sentence-transformers not available, using fallback")
            self._loaded = False
    
    def _chunk_text(self, text: str) -> List[EvidencePassage]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        char_pos = 0
        word_idx = 0
        
        while word_idx < len(words):
            # Get chunk words
            chunk_words = words[word_idx:word_idx + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            chunk_start = char_pos
            chunk_end = chunk_start + len(chunk_text)
            
            chunks.append(EvidencePassage(
                text=chunk_text,
                chunk_index=len(chunks),
                char_start=chunk_start,
                char_end=chunk_end,
            ))
            
            # Move with overlap
            word_idx += (self.chunk_size - self.chunk_overlap)
            char_pos = text.find(words[min(word_idx, len(words)-1)], char_pos) if word_idx < len(words) else chunk_end
            
            if word_idx >= len(words):
                break
        
        return chunks
    
    def _extract_claims(self, backstory_text: str) -> List[str]:
        """Extract individual claims from backstory."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', backstory_text.strip())
        
        # Filter meaningful claims
        claims = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) >= 5 and len(sent) >= 20:
                claims.append(sent)
        
        return claims
    
    def index_narrative(self, narrative_text: str):
        """
        Index narrative into chunks for evidence retrieval.
        
        Call this once per novel before checking backstories.
        """
        self.narrative_chunks = self._chunk_text(narrative_text)
        print(f"Indexed {len(self.narrative_chunks)} narrative chunks")
    
    def _score_pairs_crossencoder(
        self,
        claims: List[str],
        passages: List[str]
    ) -> torch.Tensor:
        """
        Score all claim-passage pairs using cross-encoder.
        
        Returns tensor of shape [num_claims, num_passages, 3] with 
        [contradiction, neutral, entailment] scores.
        """
        self.load()
        
        if not self._loaded or self.model is None:
            # Fallback: return uniform scores
            return torch.ones(len(claims), len(passages), 3) / 3
        
        # Create all pairs
        pairs = []
        for claim in claims:
            for passage in passages:
                pairs.append([passage, claim])  # premise, hypothesis
        
        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self.model.predict(batch, convert_to_numpy=True)
            all_scores.extend(scores)
        
        # Reshape: [claims * passages, 3] -> [claims, passages, 3]
        all_scores = torch.tensor(all_scores)
        
        # Handle 1D output (some models output single score)
        if all_scores.dim() == 1:
            # Convert similarity to 3-class
            all_scores = all_scores.unsqueeze(-1)
            # Approximate: high score = entailment, low = contradiction
            entailment = torch.sigmoid(all_scores)
            contradiction = 1 - entailment
            neutral = torch.ones_like(entailment) * 0.3
            all_scores = torch.cat([contradiction, neutral, entailment], dim=-1)
        
        all_scores = all_scores.view(len(claims), len(passages), -1)
        
        # Ensure 3 classes
        if all_scores.size(-1) != 3:
            # Pad or truncate
            if all_scores.size(-1) < 3:
                padding = torch.zeros(all_scores.size(0), all_scores.size(1), 3 - all_scores.size(-1))
                all_scores = torch.cat([all_scores, padding], dim=-1)
            else:
                all_scores = all_scores[..., :3]
        
        return all_scores
    
    def _score_pairs_fallback(
        self,
        claims: List[str],
        passages: List[str]
    ) -> torch.Tensor:
        """
        Fallback scoring using keyword matching and embedding similarity.
        Used when cross-encoder is not available.
        """
        scores = torch.zeros(len(claims), len(passages), 3)
        
        for i, claim in enumerate(claims):
            claim_words = set(claim.lower().split())
            
            for j, passage in enumerate(passages):
                passage_words = set(passage.lower().split())
                
                # Jaccard similarity as relevance
                overlap = len(claim_words & passage_words)
                union = len(claim_words | passage_words)
                similarity = overlap / max(union, 1)
                
                # Check for contradiction keywords
                has_negation = any(neg in passage.lower() for neg in ['not', 'never', "didn't", "wasn't", "no"])
                
                if similarity > 0.3:
                    if has_negation:
                        scores[i, j] = torch.tensor([0.6, 0.3, 0.1])  # Contradiction
                    else:
                        scores[i, j] = torch.tensor([0.1, 0.3, 0.6])  # Entailment
                else:
                    scores[i, j] = torch.tensor([0.2, 0.6, 0.2])  # Neutral
        
        return scores
    
    def retrieve_evidence(
        self,
        backstory_text: str,
        claims: Optional[List[str]] = None,
    ) -> EvidenceReport:
        """
        Retrieve and score evidence for backstory claims.
        
        Args:
            backstory_text: The backstory to analyze
            claims: Optional pre-extracted claims (auto-extracts if None)
            
        Returns:
            EvidenceReport with full attribution
        """
        if not self.narrative_chunks:
            raise ValueError("No narrative indexed. Call index_narrative first.")
        
        # Extract claims if not provided
        if claims is None:
            claims = self._extract_claims(backstory_text)
        
        if not claims:
            return EvidenceReport(
                claim_evidence=[],
                total_claims=0,
                is_consistent=True,
                confidence=0.5
            )
        
        print(f"Scoring {len(claims)} claims against {len(self.narrative_chunks)} passages...")
        
        # Get passage texts
        passage_texts = [p.text for p in self.narrative_chunks]
        
        # Score all pairs
        try:
            self.load()
            if self._loaded and self.model is not None:
                scores = self._score_pairs_crossencoder(claims, passage_texts)
            else:
                scores = self._score_pairs_fallback(claims, passage_texts)
        except Exception as e:
            print(f"Cross-encoder error: {e}, using fallback")
            scores = self._score_pairs_fallback(claims, passage_texts)
        
        # Apply softmax to get probabilities
        probs = F.softmax(scores, dim=-1)  # [claims, passages, 3]
        
        # Process each claim
        claim_evidences = []
        all_contradicting = []
        all_supporting = []
        
        for claim_idx, claim in enumerate(claims):
            claim_probs = probs[claim_idx]  # [passages, 3]
            
            # Sort passages by different criteria
            contradiction_scores = claim_probs[:, 0]
            entailment_scores = claim_probs[:, 2]
            
            # Create evidence passages
            supporting = []
            contradicting = []
            neutral = []
            
            for p_idx in range(len(self.narrative_chunks)):
                passage = self.narrative_chunks[p_idx]
                
                # Create evidence with scores
                evidence = EvidencePassage(
                    text=passage.text,
                    chunk_index=passage.chunk_index,
                    char_start=passage.char_start,
                    char_end=passage.char_end,
                    contradiction_score=contradiction_scores[p_idx].item(),
                    entailment_score=entailment_scores[p_idx].item(),
                    neutral_score=claim_probs[p_idx, 1].item(),
                )
                
                # Determine verdict
                max_score = max(evidence.contradiction_score, evidence.entailment_score, evidence.neutral_score)
                if evidence.contradiction_score == max_score and evidence.contradiction_score > 0.4:
                    evidence.verdict = "contradiction"
                    contradicting.append(evidence)
                elif evidence.entailment_score == max_score and evidence.entailment_score > 0.4:
                    evidence.verdict = "entailment"
                    supporting.append(evidence)
                else:
                    evidence.verdict = "neutral"
                    neutral.append(evidence)
            
            # Sort by score
            supporting.sort(key=lambda x: x.entailment_score, reverse=True)
            contradicting.sort(key=lambda x: x.contradiction_score, reverse=True)
            
            # Take top-k
            supporting = supporting[:self.top_k]
            contradicting = contradicting[:self.top_k]
            
            # Determine claim verdict
            max_contra = max([e.contradiction_score for e in contradicting], default=0)
            max_support = max([e.entailment_score for e in supporting], default=0)
            
            if max_contra > 0.6:
                verdict = "contradiction"
                confidence = max_contra
            elif max_support > 0.5:
                verdict = "entailment"
                confidence = max_support
            else:
                verdict = "neutral"
                confidence = 0.5
            
            claim_evidence = ClaimEvidence(
                claim_text=claim,
                supporting_passages=supporting,
                contradicting_passages=contradicting,
                neutral_passages=neutral[:3],
                verdict=verdict,
                confidence=confidence,
                best_supporting=supporting[0].text[:200] if supporting else None,
                best_contradicting=contradicting[0].text[:200] if contradicting else None,
            )
            claim_evidences.append(claim_evidence)
            
            # Collect for global ranking
            for e in supporting[:3]:
                all_supporting.append({
                    'claim': claim,
                    'passage': e.text[:300],
                    'score': e.entailment_score,
                    'chunk_index': e.chunk_index,
                })
            for e in contradicting[:3]:
                all_contradicting.append({
                    'claim': claim,
                    'passage': e.text[:300],
                    'score': e.contradiction_score,
                    'chunk_index': e.chunk_index,
                })
        
        # Sort global evidence
        all_supporting.sort(key=lambda x: x['score'], reverse=True)
        all_contradicting.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate summary
        supported = sum(1 for ce in claim_evidences if ce.verdict == "entailment")
        contradicted = sum(1 for ce in claim_evidences if ce.verdict == "contradiction")
        neutral = sum(1 for ce in claim_evidences if ce.verdict == "neutral")
        
        is_consistent = contradicted == 0
        
        # Overall confidence
        if contradicted > 0:
            confidence = max(ce.confidence for ce in claim_evidences if ce.verdict == "contradiction")
        else:
            confidence = sum(ce.confidence for ce in claim_evidences) / len(claim_evidences)
        
        return EvidenceReport(
            claim_evidence=claim_evidences,
            total_claims=len(claims),
            supported_claims=supported,
            contradicted_claims=contradicted,
            neutral_claims=neutral,
            top_supporting_passages=all_supporting[:self.top_k],
            top_contradicting_passages=all_contradicting[:self.top_k],
            is_consistent=is_consistent,
            confidence=confidence,
        )
    
    def generate_evidence_report(self, report: EvidenceReport) -> str:
        """
        Generate interpretable evidence report.
        
        This is the key output for interpretability requirement.
        """
        lines = []
        lines.append("# Evidence Attribution Report")
        lines.append(f"\n**Overall Verdict**: {'CONSISTENT ✅' if report.is_consistent else 'CONTRADICTION ❌'}")
        lines.append(f"**Confidence**: {report.confidence:.1%}")
        
        lines.append("\n## Claim Analysis")
        lines.append(f"- Total claims: {report.total_claims}")
        lines.append(f"- Supported: {report.supported_claims}")
        lines.append(f"- Contradicted: {report.contradicted_claims}")
        lines.append(f"- Neutral: {report.neutral_claims}")
        
        # Contradicting claims (most important)
        if report.contradicted_claims > 0:
            lines.append("\n## ⚠️ Contradicting Claims")
            for ce in report.claim_evidence:
                if ce.verdict == "contradiction":
                    lines.append(f"\n### Claim: \"{ce.claim_text}\"")
                    lines.append(f"**Confidence**: {ce.confidence:.1%}")
                    if ce.best_contradicting:
                        lines.append(f"\n**Contradicting Passage**:")
                        lines.append(f"> \"{ce.best_contradicting}...\"")
        
        # Top contradicting evidence
        if report.top_contradicting_passages:
            lines.append("\n## Key Contradicting Evidence")
            for i, ev in enumerate(report.top_contradicting_passages[:3], 1):
                lines.append(f"\n### Evidence {i} (score: {ev['score']:.2f})")
                lines.append(f"**Claim**: \"{ev['claim']}\"")
                lines.append(f"**Passage [Chunk {ev['chunk_index']}]**:")
                lines.append(f"> \"{ev['passage']}...\"")
        
        # Supporting evidence
        if report.top_supporting_passages:
            lines.append("\n## Supporting Evidence")
            for i, ev in enumerate(report.top_supporting_passages[:3], 1):
                lines.append(f"\n### Evidence {i} (score: {ev['score']:.2f})")
                lines.append(f"**Claim**: \"{ev['claim']}\"")
                lines.append(f"**Passage [Chunk {ev['chunk_index']}]**:")
                lines.append(f"> \"{ev['passage'][:150]}...\"")
        
        return "\n".join(lines)
