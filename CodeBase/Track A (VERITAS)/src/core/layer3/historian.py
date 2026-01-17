"""
Historian: Evidence-based verification stream.

The Historian is a cold, logical fact-checker that classifies retrieved
evidence as Supporting, Contradicting, or Missing.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from kdsh.layer1.constraint_graph import Constraint
from kdsh.layer2.unified_retriever import Evidence, RetrievalResult
from kdsh.models import llm_complete


class EvidenceStatus(str, Enum):
    """Status of evidence relative to a constraint."""
    
    SUPPORTING = "SUPPORTING"
    CONTRADICTING = "CONTRADICTING"  
    NEUTRAL = "NEUTRAL"
    MISSING = "MISSING"


@dataclass
class ClassifiedEvidence:
    """Evidence with its classification."""
    
    evidence: Evidence
    status: EvidenceStatus
    weight: float
    reasoning: str = ""


@dataclass
class HistorianVerdict:
    """Verdict from the Historian verification stream."""
    
    constraint_id: str
    status: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "UNKNOWN"]
    confidence: float
    supporting_evidence: list[ClassifiedEvidence] = field(default_factory=list)
    contradicting_evidence: list[ClassifiedEvidence] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    reasoning: str = ""
    
    @property
    def has_contradiction(self) -> bool:
        return self.status == "CONTRADICTED"
    
    @property
    def max_contradiction_weight(self) -> float:
        if not self.contradicting_evidence:
            return 0.0
        return max(e.weight for e in self.contradicting_evidence)


CLASSIFICATION_PROMPT = """You are a meticulous fact-checker analyzing whether evidence supports or contradicts a claim.

CLAIM TO VERIFY:
{claim}

EVIDENCE TEXT:
{evidence}

Analyze the relationship between the claim and evidence. Consider:
1. Does the evidence directly confirm the claim?
2. Does the evidence contradict or negate the claim?
3. Is the evidence neutral/irrelevant?

Output a JSON object with:
- "status": One of "SUPPORTING", "CONTRADICTING", or "NEUTRAL"
- "weight": Confidence from 0.0 to 1.0
- "reasoning": Brief explanation

Output ONLY valid JSON:"""


MISSING_EVIDENCE_PROMPT = """Given a claim, what evidence would we EXPECT to find in a novel if this claim were true?

CLAIM:
{claim}

List 2-3 specific pieces of evidence we would expect to find. Output as a JSON array of strings:"""


class Historian:
    """
    Evidence-based verification stream.
    
    The Historian classifies each piece of retrieved evidence as:
    - SUPPORTING: Evidence that confirms the constraint
    - CONTRADICTING: Evidence that negates the constraint  
    - NEUTRAL: Evidence that is irrelevant
    - MISSING: Expected evidence that was not found
    """
    
    def __init__(self, model: str | None = None):
        """
        Initialize the Historian.
        
        Args:
            model: Optional model override
        """
        self.model = model
    
    def verify(
        self, 
        constraint: Constraint, 
        retrieval_result: RetrievalResult
    ) -> HistorianVerdict:
        """
        Verify a constraint against retrieved evidence.
        
        Args:
            constraint: The constraint to verify
            retrieval_result: Evidence from retrieval
            
        Returns:
            HistorianVerdict with classification of all evidence
        """
        supporting = []
        contradicting = []
        
        # Classify each piece of evidence
        for evidence in retrieval_result.raptor_evidence:
            classified = self._classify_evidence(constraint.claim, evidence)
            
            if classified.status == EvidenceStatus.SUPPORTING:
                supporting.append(classified)
            elif classified.status == EvidenceStatus.CONTRADICTING:
                contradicting.append(classified)
        
        # Check for missing evidence
        missing = self._check_missing_evidence(constraint, retrieval_result)
        
        # Determine overall verdict
        status, confidence, reasoning = self._determine_verdict(
            constraint, supporting, contradicting, missing
        )
        
        return HistorianVerdict(
            constraint_id=constraint.id,
            status=status,
            confidence=confidence,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            missing_evidence=missing,
            reasoning=reasoning
        )
    
    def _classify_evidence(
        self, 
        claim: str, 
        evidence: Evidence
    ) -> ClassifiedEvidence:
        """Classify a single piece of evidence."""
        prompt = CLASSIFICATION_PROMPT.format(
            claim=claim,
            evidence=evidence.text[:2000]
        )
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="You are an objective fact-checker. Output only valid JSON.",
            model=self.model,
            temperature=0.0,
            max_tokens=300
        )
        
        # Parse response
        import json
        import re
        
        try:
            # Clean response
            response = response.strip()
            if "```json" in response:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            
            data = json.loads(response)
            status = EvidenceStatus(data.get("status", "NEUTRAL"))
            weight = float(data.get("weight", 0.5))
            reasoning = data.get("reasoning", "")
            
        except (json.JSONDecodeError, ValueError, KeyError):
            status = EvidenceStatus.NEUTRAL
            weight = 0.5
            reasoning = "Could not classify"
        
        return ClassifiedEvidence(
            evidence=evidence,
            status=status,
            weight=weight,
            reasoning=reasoning
        )
    
    def _check_missing_evidence(
        self,
        constraint: Constraint,
        retrieval_result: RetrievalResult
    ) -> list[str]:
        """Check for expected evidence that is missing."""
        # Get expected evidence
        prompt = MISSING_EVIDENCE_PROMPT.format(claim=constraint.claim)
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="Output only a JSON array of strings.",
            model=self.model,
            temperature=0.0,
            max_tokens=300
        )
        
        # Parse expected evidence
        import json
        import re
        
        try:
            response = response.strip()
            if "```" in response:
                match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            
            expected = json.loads(response)
            if not isinstance(expected, list):
                expected = []
        except json.JSONDecodeError:
            expected = []
        
        # Check which expected items are missing
        missing = []
        evidence_texts = " ".join(retrieval_result.get_all_evidence_texts()).lower()
        
        for item in expected:
            # Simple check: is key term present?
            key_terms = item.lower().split()[:3]  # First 3 words
            if not any(term in evidence_texts for term in key_terms):
                missing.append(item)
        
        return missing
    
    def _determine_verdict(
        self,
        constraint: Constraint,
        supporting: list[ClassifiedEvidence],
        contradicting: list[ClassifiedEvidence],
        missing: list[str]
    ) -> tuple[str, float, str]:
        """Determine overall verdict from classified evidence."""
        # Strong contradiction
        if contradicting:
            max_weight = max(e.weight for e in contradicting)
            if max_weight >= 0.8:
                return (
                    "CONTRADICTED",
                    max_weight,
                    f"Found {len(contradicting)} contradicting evidence with max weight {max_weight:.2f}"
                )
        
        # Strong support
        if supporting and not contradicting:
            avg_weight = sum(e.weight for e in supporting) / len(supporting)
            if avg_weight >= 0.7 and len(supporting) >= 2:
                return (
                    "SUPPORTED", 
                    avg_weight,
                    f"Found {len(supporting)} supporting evidence with avg weight {avg_weight:.2f}"
                )
        
        # Insufficient evidence
        if not supporting and not contradicting:
            if missing:
                return (
                    "INSUFFICIENT",
                    0.5,
                    f"No evidence found. {len(missing)} expected items missing."
                )
            return ("UNKNOWN", 0.3, "No clear evidence found")
        
        # Mixed signals
        if supporting and contradicting:
            supp_weight = sum(e.weight for e in supporting)
            cont_weight = sum(e.weight for e in contradicting)
            
            if cont_weight > supp_weight:
                return (
                    "CONTRADICTED",
                    cont_weight / (supp_weight + cont_weight),
                    "Mixed evidence, contradictions outweigh support"
                )
            else:
                return (
                    "SUPPORTED",
                    supp_weight / (supp_weight + cont_weight),
                    "Mixed evidence, support outweighs contradictions"
                )
        
        return ("UNKNOWN", 0.5, "Unable to determine")
