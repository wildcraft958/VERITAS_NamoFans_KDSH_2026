"""
Conservative Adjudicator: Final binary decision using asymmetric rules.

Applies 5 decision rules with the principle that it's easier to prove
false than true. One hard contradiction is fatal.
"""

from dataclasses import dataclass, field
from typing import Literal

from core.config import settings
from core.layer4.ledger import LedgerEntry


@dataclass
class FinalVerdict:
    """
    Final verdict for a backstory.
    
    Includes the binary prediction, confidence score, and
    detailed reasoning chain.
    """
    
    prediction: Literal[0, 1]  # 0 = CONTRADICTORY, 1 = CONSISTENT
    confidence: float
    verdict_label: Literal["CONTRADICTORY", "CONSISTENT"]
    
    # Reasoning
    reasoning_chain: list[str] = field(default_factory=list)
    key_contradictions: list[dict] = field(default_factory=list)
    
    # Evidence summary
    supporting_count: int = 0
    contradicting_count: int = 0
    missing_count: int = 0
    
    # Rule that triggered the decision
    triggered_rule: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "verdict": self.verdict_label,
            "reasoning": self.reasoning_chain,
            "key_contradictions": self.key_contradictions,
            "evidence_summary": {
                "supporting_count": self.supporting_count,
                "contradicting_count": self.contradicting_count,
                "missing_count": self.missing_count
            },
            "triggered_rule": self.triggered_rule
        }


class ConservativeAdjudicator:
    """
    Conservative adjudicator using 5 asymmetric decision rules.
    
    Rules are applied sequentially - first matching rule determines
    the verdict. The rules are biased toward rejection (conservative)
    because it's easier to prove a claim false than true.
    
    Rules:
    1. HARD CONTRADICTION: Any contradiction weight > threshold → REJECT
    2. MULTIPLE MISSING: ≥2 high-impact missing evidence → REJECT  
    3. LOW PERSONA ALIGNMENT: Simulator alignment < threshold → REJECT
    4. STRONG SUPPORT: ≥3 supporting + high alignment → ACCEPT
    5. DEFAULT: Conservative rejection
    """
    
    def __init__(
        self,
        contradiction_threshold: float | None = None,
        persona_threshold: float | None = None,
        support_count_threshold: int | None = None
    ):
        """
        Initialize the adjudicator.
        
        Args:
            contradiction_threshold: Override for contradiction weight threshold
            persona_threshold: Override for persona alignment threshold
            support_count_threshold: Override for support count threshold
        """
        self.contradiction_threshold = (
            contradiction_threshold or 
            settings.adjudication.contradiction_threshold
        )
        self.persona_threshold = (
            persona_threshold or 
            settings.adjudication.persona_alignment_threshold
        )
        self.support_count_threshold = (
            support_count_threshold or 
            settings.adjudication.support_count_threshold
        )
    
    def decide(self, ledger_entries: list[LedgerEntry]) -> FinalVerdict:
        """
        Make final binary decision based on all ledger entries.
        
        Args:
            ledger_entries: List of evidence ledger entries
            
        Returns:
            FinalVerdict with prediction and reasoning
        """
        reasoning = []
        key_contradictions = []
        
        # Aggregate statistics
        total_supporting = 0
        total_contradicting = 0
        total_missing = 0
        hard_contradictions = []
        low_alignment_entries = []
        
        for entry in ledger_entries:
            total_supporting += len(entry.supporting_evidence)
            total_contradicting += len(entry.contradicting_evidence)
            total_missing += len([m for m in entry.missing_evidence if not m.found])
            
            # Check for hard contradictions
            for evidence in entry.contradicting_evidence:
                if evidence.weight >= self.contradiction_threshold:
                    hard_contradictions.append({
                        "constraint": entry.constraint_id,
                        "evidence": evidence.text[:200],
                        "weight": evidence.weight,
                        "severity": evidence.severity
                    })
            
            # Check persona alignment
            if entry.simulator_verdict.get("alignment", 1.0) < self.persona_threshold:
                low_alignment_entries.append(entry)
        
        reasoning.append(
            f"Analyzed {len(ledger_entries)} constraints with "
            f"{total_supporting} supporting, {total_contradicting} contradicting, "
            f"and {total_missing} missing evidence signals"
        )
        
        # Rule 1: HARD CONTRADICTION
        if hard_contradictions:
            reasoning.append(
                f"Rule 1 applies: Found {len(hard_contradictions)} hard contradictions "
                f"(weight ≥ {self.contradiction_threshold})"
            )
            
            return FinalVerdict(
                prediction=0,
                confidence=0.9,
                verdict_label="CONTRADICTORY",
                reasoning_chain=reasoning,
                key_contradictions=hard_contradictions[:5],
                supporting_count=total_supporting,
                contradicting_count=total_contradicting,
                missing_count=total_missing,
                triggered_rule="Rule 1: Hard Contradiction"
            )
        
        # Rule 2: MULTIPLE MISSING
        high_impact_missing = sum(
            1 for entry in ledger_entries
            for m in entry.missing_evidence
            if m.impact in ("CRITICAL", "HIGH") and not m.found
        )
        
        if high_impact_missing >= 2:
            reasoning.append(
                f"Rule 2 applies: {high_impact_missing} high-impact evidence items missing"
            )
            
            return FinalVerdict(
                prediction=0,
                confidence=0.7,
                verdict_label="CONTRADICTORY",
                reasoning_chain=reasoning,
                key_contradictions=[
                    {"type": "missing_evidence", "count": high_impact_missing}
                ],
                supporting_count=total_supporting,
                contradicting_count=total_contradicting,
                missing_count=total_missing,
                triggered_rule="Rule 2: Multiple Missing Evidence"
            )
        
        # Rule 3: LOW PERSONA ALIGNMENT
        if low_alignment_entries:
            avg_alignment = sum(
                e.simulator_verdict.get("alignment", 0.5) 
                for e in low_alignment_entries
            ) / len(low_alignment_entries)
            
            reasoning.append(
                f"Rule 3 applies: {len(low_alignment_entries)} constraints with "
                f"persona alignment < {self.persona_threshold} (avg: {avg_alignment:.2f})"
            )
            
            return FinalVerdict(
                prediction=0,
                confidence=0.75,
                verdict_label="CONTRADICTORY",
                reasoning_chain=reasoning,
                key_contradictions=[
                    {
                        "constraint": e.constraint_id,
                        "type": "persona_mismatch",
                        "alignment": e.simulator_verdict.get("alignment", 0)
                    }
                    for e in low_alignment_entries[:3]
                ],
                supporting_count=total_supporting,
                contradicting_count=total_contradicting,
                missing_count=total_missing,
                triggered_rule="Rule 3: Low Persona Alignment"
            )
        
        # Rule 4: STRONG SUPPORT
        if (total_supporting >= self.support_count_threshold and 
            total_contradicting == 0 and
            all(
                entry.simulator_verdict.get("alignment", 0) >= 0.7
                for entry in ledger_entries
                if entry.simulator_verdict
            )):
            
            avg_alignment = sum(
                entry.simulator_verdict.get("alignment", 0.5)
                for entry in ledger_entries
            ) / max(len(ledger_entries), 1)
            
            reasoning.append(
                f"Rule 4 applies: {total_supporting} supporting evidence, "
                f"no contradictions, avg persona alignment {avg_alignment:.2f}"
            )
            
            return FinalVerdict(
                prediction=1,
                confidence=0.8,
                verdict_label="CONSISTENT",
                reasoning_chain=reasoning,
                key_contradictions=[],
                supporting_count=total_supporting,
                contradicting_count=total_contradicting,
                missing_count=total_missing,
                triggered_rule="Rule 4: Strong Support"
            )
        
        # Rule 5: DEFAULT (Conservative)
        reasoning.append(
            "Rule 5 applies: Insufficient evidence for acceptance (conservative default)"
        )
        
        return FinalVerdict(
            prediction=0,
            confidence=0.5,
            verdict_label="CONTRADICTORY",
            reasoning_chain=reasoning,
            key_contradictions=[],
            supporting_count=total_supporting,
            contradicting_count=total_contradicting,
            missing_count=total_missing,
            triggered_rule="Rule 5: Default Conservative"
        )
    
    def explain_decision(self, verdict: FinalVerdict) -> str:
        """Generate human-readable explanation of the decision."""
        lines = [
            f"VERDICT: {verdict.verdict_label} (prediction={verdict.prediction})",
            f"Confidence: {verdict.confidence:.2f}",
            f"Triggered by: {verdict.triggered_rule}",
            "",
            "Reasoning Chain:",
        ]
        
        for i, reason in enumerate(verdict.reasoning_chain, 1):
            lines.append(f"  {i}. {reason}")
        
        if verdict.key_contradictions:
            lines.append("")
            lines.append("Key Contradictions:")
            for i, cont in enumerate(verdict.key_contradictions[:3], 1):
                if "evidence" in cont:
                    lines.append(f"  {i}. {cont.get('evidence', '')[:100]}...")
                elif "type" in cont:
                    lines.append(f"  {i}. {cont['type']}: {cont}")
        
        lines.append("")
        lines.append(
            f"Evidence Summary: {verdict.supporting_count} supporting, "
            f"{verdict.contradicting_count} contradicting, "
            f"{verdict.missing_count} missing"
        )
        
        return "\n".join(lines)
