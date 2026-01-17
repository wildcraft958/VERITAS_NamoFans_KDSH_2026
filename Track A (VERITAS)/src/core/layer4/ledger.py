"""
Evidence Ledger: Structured tracking of all evidence.

Aggregates signals from Historian and Simulator into a structured
JSON ledger for each constraint.
"""

from dataclasses import dataclass, field
from typing import Literal

from kdsh.layer1.constraint_graph import Constraint
from kdsh.layer3.historian import HistorianVerdict, ClassifiedEvidence
from kdsh.layer3.simulator import SimulatorVerdict


@dataclass
class EvidenceEntry:
    """A classified piece of evidence for the ledger."""
    
    text: str
    source: str
    weight: float
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    from_raptor: bool = True
    from_hipporag: bool = False
    rationale: str = ""


@dataclass
class MissingEvidence:
    """Expected evidence that was not found."""
    
    expected: str
    found: bool = False
    impact: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = "MEDIUM"


@dataclass
class LedgerEntry:
    """
    Complete evidence ledger entry for a single constraint.
    
    Aggregates all evidence (supporting, contradicting, missing)
    and verdicts from both Historian and Simulator.
    """
    
    constraint_id: str
    claim: str
    constraint_type: str
    
    # Evidence
    supporting_evidence: list[EvidenceEntry] = field(default_factory=list)
    contradicting_evidence: list[EvidenceEntry] = field(default_factory=list)
    missing_evidence: list[MissingEvidence] = field(default_factory=list)
    
    # Verdicts
    historian_verdict: dict = field(default_factory=dict)
    simulator_verdict: dict = field(default_factory=dict)
    
    # Final assessment
    final_verdict: Literal["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "UNKNOWN"] = "UNKNOWN"
    confidence: float = 0.5
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "constraint_id": self.constraint_id,
            "claim": self.claim,
            "type": self.constraint_type,
            "supporting_evidence": [
                {
                    "text": e.text[:200],
                    "source": e.source,
                    "weight": e.weight,
                    "severity": e.severity
                }
                for e in self.supporting_evidence
            ],
            "contradicting_evidence": [
                {
                    "text": e.text[:200],
                    "source": e.source,
                    "weight": e.weight,
                    "severity": e.severity
                }
                for e in self.contradicting_evidence
            ],
            "missing_evidence": [
                {"expected": m.expected, "found": m.found, "impact": m.impact}
                for m in self.missing_evidence
            ],
            "historian_verdict": self.historian_verdict,
            "simulator_verdict": self.simulator_verdict,
            "final_verdict": self.final_verdict,
            "confidence": self.confidence
        }


class EvidenceLedger:
    """
    Aggregates evidence from Historian and Simulator into structured ledger.
    
    The ledger provides:
    - Explicit tracking of supporting evidence
    - Explicit tracking of contradicting evidence  
    - Explicit tracking of MISSING evidence (absence-of-evidence signals)
    - Combined verdicts from both verification streams
    """
    
    def __init__(self):
        """Initialize the evidence ledger."""
        self.entries: dict[str, LedgerEntry] = {}
    
    def aggregate(
        self,
        constraint: Constraint,
        historian_verdict: HistorianVerdict,
        simulator_verdict: SimulatorVerdict
    ) -> LedgerEntry:
        """
        Aggregate evidence for a constraint into a ledger entry.
        
        Args:
            constraint: The constraint being verified
            historian_verdict: Verdict from Historian
            simulator_verdict: Verdict from Simulator
            
        Returns:
            LedgerEntry with all aggregated evidence
        """
        entry = LedgerEntry(
            constraint_id=constraint.id,
            claim=constraint.claim,
            constraint_type=constraint.type.value
        )
        
        # Process supporting evidence from Historian
        for classified in historian_verdict.supporting_evidence:
            severity = self._weight_to_severity(classified.weight)
            evidence_entry = EvidenceEntry(
                text=classified.evidence.text,
                source=classified.evidence.source,
                weight=classified.weight,
                severity=severity,
                rationale=classified.reasoning
            )
            entry.supporting_evidence.append(evidence_entry)
        
        # Process contradicting evidence from Historian
        for classified in historian_verdict.contradicting_evidence:
            severity = self._weight_to_severity(classified.weight)
            evidence_entry = EvidenceEntry(
                text=classified.evidence.text,
                source=classified.evidence.source,
                weight=classified.weight,
                severity=severity,
                rationale=classified.reasoning
            )
            entry.contradicting_evidence.append(evidence_entry)
        
        # Process missing evidence
        for missing in historian_verdict.missing_evidence:
            impact = "HIGH" if any(
                kw in missing.lower() 
                for kw in ["death", "escape", "critical", "essential"]
            ) else "MEDIUM"
            
            entry.missing_evidence.append(MissingEvidence(
                expected=missing,
                found=False,
                impact=impact
            ))
        
        # Add out-of-character moments as contradicting signals
        for moment in simulator_verdict.out_of_character_moments:
            entry.contradicting_evidence.append(EvidenceEntry(
                text=moment,
                source="Simulator persona analysis",
                weight=0.7,
                severity="HIGH",
                rationale="Out-of-character behavior detected"
            ))
        
        # Store verdicts
        entry.historian_verdict = {
            "status": historian_verdict.status,
            "confidence": historian_verdict.confidence,
            "reasoning": historian_verdict.reasoning
        }
        
        entry.simulator_verdict = {
            "status": simulator_verdict.status,
            "alignment": simulator_verdict.persona_alignment,
            "reasoning": simulator_verdict.reasoning
        }
        
        # Determine final verdict
        entry.final_verdict, entry.confidence = self._combine_verdicts(
            historian_verdict, simulator_verdict, entry
        )
        
        # Cache entry
        self.entries[constraint.id] = entry
        
        return entry
    
    def _weight_to_severity(self, weight: float) -> Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        """Convert weight to severity level."""
        if weight >= 0.9:
            return "CRITICAL"
        elif weight >= 0.7:
            return "HIGH"
        elif weight >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _combine_verdicts(
        self,
        historian: HistorianVerdict,
        simulator: SimulatorVerdict,
        entry: LedgerEntry
    ) -> tuple[str, float]:
        """Combine Historian and Simulator verdicts."""
        # Priority: Contradictions take precedence
        
        # Strong contradiction from either
        if historian.status == "CONTRADICTED" and historian.confidence >= 0.8:
            return "CONTRADICTED", historian.confidence
        
        if simulator.status == "OUT_OF_CHARACTER" and simulator.persona_alignment <= 0.3:
            return "CONTRADICTED", 1.0 - simulator.persona_alignment
        
        # Both support
        if historian.status == "SUPPORTED" and simulator.status == "IN_CHARACTER":
            combined_conf = (historian.confidence + simulator.persona_alignment) / 2
            return "SUPPORTED", combined_conf
        
        # Mixed signals - lean toward contradiction
        if historian.status == "CONTRADICTED" or simulator.status == "OUT_OF_CHARACTER":
            return "CONTRADICTED", 0.7
        
        # Insufficient evidence
        if historian.status == "INSUFFICIENT":
            return "INSUFFICIENT", 0.5
        
        return "UNKNOWN", 0.5
    
    def get_all_entries(self) -> list[LedgerEntry]:
        """Get all ledger entries."""
        return list(self.entries.values())
    
    def get_contradicted_entries(self) -> list[LedgerEntry]:
        """Get entries with CONTRADICTED verdict."""
        return [e for e in self.entries.values() if e.final_verdict == "CONTRADICTED"]
    
    def get_supported_entries(self) -> list[LedgerEntry]:
        """Get entries with SUPPORTED verdict."""
        return [e for e in self.entries.values() if e.final_verdict == "SUPPORTED"]
    
    def to_json(self) -> list[dict]:
        """Convert all entries to JSON-serializable format."""
        return [entry.to_dict() for entry in self.entries.values()]
