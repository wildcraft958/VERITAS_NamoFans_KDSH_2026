"""
Document-Level NLI Classification using DeBERTa-v3

SOTA approach for detecting entailment/contradiction at document scale.
Uses sliding window with proper aggregation for long narratives.

Key features:
- DeBERTa-v3-large-mnli-fever backbone (proven SOTA on NLI)
- Sliding window for 100k+ word documents
- Claim-level granularity for interpretability
- Returns passage citations for each verdict
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import re


@dataclass
class ClaimScore:
    """Score for a single claim against the narrative."""
    claim_text: str
    label: str  # "entailment", "contradiction", "neutral"
    confidence: float
    
    # Evidence for interpretability
    supporting_passages: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_passages: List[Dict[str, Any]] = field(default_factory=list)
    
    def is_contradiction(self) -> bool:
        return self.label == "contradiction"


@dataclass
class DocumentNLIResult:
    """Result of document-level NLI."""
    consistent: bool
    confidence: float
    
    # Claim-level breakdown for interpretability
    claim_scores: List[ClaimScore] = field(default_factory=list)
    
    # Aggregate statistics
    num_contradictions: int = 0
    num_entailments: int = 0
    num_neutral: int = 0
    
    # Key evidence passages
    top_contradicting_passages: List[Dict[str, Any]] = field(default_factory=list)
    top_supporting_passages: List[Dict[str, Any]] = field(default_factory=list)


class DocumentNLIClassifier:
    """
    SOTA Document-level NLI using DeBERTa-v3 with interpretable outputs.
    
    Architecture:
    1. Parse backstory into individual claims
    2. Chunk narrative into overlapping passages
    3. Score each claim against each relevant passage
    4. Aggregate with max-pooling for contradictions (conservative)
    5. Return passage-level evidence for interpretability
    
    Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    - Trained on: MNLI, FEVER, ANLI, LingNLI, WANLI
    - Best open-source NLI model as of 2024
    """
    
    LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}
    
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        device: torch.device = None,
        chunk_size: int = 256,  # tokens per chunk
        chunk_overlap: int = 64,  # overlap between chunks
        batch_size: int = 8,
        contradiction_threshold: float = 0.7,  # confidence threshold
    ):
        self.model_name = model_name
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.contradiction_threshold = contradiction_threshold
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
    def load(self):
        """Lazy load the model (saves memory if not used)."""
        if self._loaded:
            return
            
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        print(f"Loading NLI model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"NLI model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        self._loaded = True
    
    def _extract_claims(self, backstory_text: str) -> List[str]:
        """
        Extract individual claims from backstory text.
        
        Each sentence is treated as a potential claim.
        Filter out very short sentences (< 5 words).
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', backstory_text.strip())
        
        # Filter and clean
        claims = []
        for sent in sentences:
            sent = sent.strip()
            words = sent.split()
            if len(words) >= 5 and len(sent) >= 20:
                claims.append(sent)
        
        return claims
    
    def _chunk_narrative(self, narrative_text: str) -> List[Dict[str, Any]]:
        """
        Chunk narrative into overlapping passages for processing.
        
        Returns list of dicts with:
        - text: The passage text
        - start_char: Starting character position
        - end_char: Ending character position
        """
        self.load()
        
        # Tokenize to get consistent chunks
        tokens = self.tokenizer.encode(narrative_text, add_special_tokens=False)
        
        chunks = []
        token_idx = 0
        
        while token_idx < len(tokens):
            end_idx = min(token_idx + self.chunk_size, len(tokens))
            chunk_tokens = tokens[token_idx:end_idx]
            
            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            
            # Approximate character positions (rough estimate for citation)
            words_before = len(self.tokenizer.decode(tokens[:token_idx], skip_special_tokens=True))
            
            chunks.append({
                'text': chunk_text,
                'token_start': token_idx,
                'token_end': end_idx,
                'chunk_index': len(chunks),
                'approx_char_start': words_before,
            })
            
            # Move with overlap
            token_idx += (self.chunk_size - self.chunk_overlap)
            
            if token_idx >= end_idx:
                break
        
        return chunks
    
    def _score_claim_passage_batch(
        self, 
        claims: List[str], 
        passages: List[str]
    ) -> torch.Tensor:
        """
        Score all claim-passage pairs.
        
        Returns tensor of shape [num_claims, num_passages, 3] with logits.
        """
        self.load()
        
        # Create all pairs
        pairs = []
        for claim in claims:
            for passage in passages:
                pairs.append((passage, claim))  # NLI format: premise, hypothesis
        
        all_logits = []
        
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i:i + self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                [p[0] for p in batch_pairs],  # premises
                [p[1] for p in batch_pairs],  # hypotheses
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # [batch, 3]
                all_logits.append(logits.cpu())
        
        # Concatenate and reshape
        all_logits = torch.cat(all_logits, dim=0)  # [num_claims * num_passages, 3]
        all_logits = all_logits.view(len(claims), len(passages), 3)
        
        return all_logits
    
    def classify(
        self,
        narrative_text: str,
        backstory_text: str,
        return_all_evidence: bool = False,
        top_k_evidence: int = 5,
    ) -> DocumentNLIResult:
        """
        Classify backstory consistency with full interpretability.
        
        Args:
            narrative_text: The novel text (can be 100k+ words)
            backstory_text: The hypothetical backstory
            return_all_evidence: If True, return all passages, not just top-k
            top_k_evidence: Number of evidence passages to return per claim
            
        Returns:
            DocumentNLIResult with claim-level breakdown and evidence
        """
        self.load()
        
        # 1. Extract claims from backstory
        claims = self._extract_claims(backstory_text)
        if not claims:
            # No claims extracted, treat as neutral
            return DocumentNLIResult(
                consistent=True,
                confidence=0.5,
                claim_scores=[],
            )
        
        print(f"Extracted {len(claims)} claims from backstory")
        
        # 2. Chunk narrative
        chunks = self._chunk_narrative(narrative_text)
        print(f"Created {len(chunks)} narrative chunks")
        
        if not chunks:
            return DocumentNLIResult(
                consistent=True,
                confidence=0.5,
                claim_scores=[],
            )
        
        # 3. Score all claim-passage pairs
        print("Scoring claim-passage pairs...")
        passage_texts = [c['text'] for c in chunks]
        logits = self._score_claim_passage_batch(claims, passage_texts)
        probs = F.softmax(logits, dim=-1)  # [claims, passages, 3]
        
        # 4. Process each claim
        claim_scores = []
        all_contradicting = []
        all_supporting = []
        
        for claim_idx, claim in enumerate(claims):
            claim_probs = probs[claim_idx]  # [passages, 3]
            
            # Get max contradiction and entailment scores across all passages
            contradiction_scores = claim_probs[:, 0]  # contradiction
            entailment_scores = claim_probs[:, 2]  # entailment
            
            max_contradiction_score = contradiction_scores.max().item()
            max_entailment_score = entailment_scores.max().item()
            
            # Determine claim label
            if max_contradiction_score > self.contradiction_threshold:
                label = "contradiction"
                confidence = max_contradiction_score
            elif max_entailment_score > 0.6:
                label = "entailment"
                confidence = max_entailment_score
            else:
                label = "neutral"
                confidence = claim_probs[:, 1].max().item()
            
            # Get top passages for evidence
            supporting_passages = []
            contradicting_passages = []
            
            # Top contradicting passages
            top_contra_indices = contradiction_scores.argsort(descending=True)[:top_k_evidence]
            for idx in top_contra_indices:
                idx = idx.item()
                if contradiction_scores[idx] > 0.3:  # Minimum threshold
                    passage_info = {
                        'chunk_index': chunks[idx]['chunk_index'],
                        'text': chunks[idx]['text'][:500],  # Truncate for display
                        'score': contradiction_scores[idx].item(),
                        'label': 'contradiction',
                    }
                    contradicting_passages.append(passage_info)
                    all_contradicting.append({
                        **passage_info,
                        'claim': claim,
                    })
            
            # Top supporting passages
            top_support_indices = entailment_scores.argsort(descending=True)[:top_k_evidence]
            for idx in top_support_indices:
                idx = idx.item()
                if entailment_scores[idx] > 0.3:
                    passage_info = {
                        'chunk_index': chunks[idx]['chunk_index'],
                        'text': chunks[idx]['text'][:500],
                        'score': entailment_scores[idx].item(),
                        'label': 'entailment',
                    }
                    supporting_passages.append(passage_info)
                    all_supporting.append({
                        **passage_info,
                        'claim': claim,
                    })
            
            claim_scores.append(ClaimScore(
                claim_text=claim,
                label=label,
                confidence=confidence,
                supporting_passages=supporting_passages,
                contradicting_passages=contradicting_passages,
            ))
        
        # 5. Aggregate to document level
        num_contradictions = sum(1 for cs in claim_scores if cs.label == "contradiction")
        num_entailments = sum(1 for cs in claim_scores if cs.label == "entailment")
        num_neutral = sum(1 for cs in claim_scores if cs.label == "neutral")
        
        # Conservative: any high-confidence contradiction means inconsistent
        has_contradiction = num_contradictions > 0
        
        # Confidence based on claim distribution
        if has_contradiction:
            # Take max contradiction confidence
            confidence = max(cs.confidence for cs in claim_scores if cs.label == "contradiction")
        else:
            # Average confidence
            confidence = sum(cs.confidence for cs in claim_scores) / len(claim_scores)
        
        # Sort evidence by score
        all_contradicting.sort(key=lambda x: x['score'], reverse=True)
        all_supporting.sort(key=lambda x: x['score'], reverse=True)
        
        return DocumentNLIResult(
            consistent=not has_contradiction,
            confidence=confidence,
            claim_scores=claim_scores,
            num_contradictions=num_contradictions,
            num_entailments=num_entailments,
            num_neutral=num_neutral,
            top_contradicting_passages=all_contradicting[:top_k_evidence],
            top_supporting_passages=all_supporting[:top_k_evidence],
        )
    
    def generate_rationale(self, result: DocumentNLIResult) -> str:
        """
        Generate human-readable rationale from NLI result.
        
        For interpretability - explains WHY the verdict was reached.
        """
        lines = []
        
        if result.consistent:
            lines.append("## NLI Analysis: CONSISTENT")
        else:
            lines.append("## NLI Analysis: CONTRADICTION DETECTED")
        
        lines.append(f"\n**Confidence**: {result.confidence:.1%}")
        lines.append(f"\n**Claim Breakdown**:")
        lines.append(f"- Contradictions: {result.num_contradictions}")
        lines.append(f"- Entailments: {result.num_entailments}")
        lines.append(f"- Neutral: {result.num_neutral}")
        
        # Show contradicting claims with evidence
        if result.num_contradictions > 0:
            lines.append("\n### Contradicting Claims:")
            for cs in result.claim_scores:
                if cs.label == "contradiction":
                    lines.append(f"\n**Claim**: \"{cs.claim_text}\"")
                    lines.append(f"  - Confidence: {cs.confidence:.1%}")
                    if cs.contradicting_passages:
                        lines.append("  - **Evidence**:")
                        for passage in cs.contradicting_passages[:2]:
                            lines.append(f"    - [Chunk {passage['chunk_index']}] \"{passage['text'][:150]}...\"")
                            lines.append(f"      (contradiction score: {passage['score']:.2f})")
        
        # Show supporting evidence
        if result.num_entailments > 0:
            lines.append("\n### Supported Claims:")
            for cs in result.claim_scores:
                if cs.label == "entailment":
                    lines.append(f"\n**Claim**: \"{cs.claim_text}\"")
                    lines.append(f"  - Confidence: {cs.confidence:.1%}")
        
        return "\n".join(lines)
