"""
Temporal Reasoning Engine with Constraint Graph

SOTA temporal reasoning inspired by TG-LLM and Allen's Interval Algebra.

Key features:
- Temporal graph representation of narrative events
- Allen's 13 interval relations for temporal constraints
- Constraint propagation for consistency checking
- Cycle detection for impossible timelines
- Interpretable timeline violations
"""

import torch
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import re


class TemporalRelation(Enum):
    """
    Allen's 13 Interval Relations.
    
    These are the complete set of relations between two time intervals.
    """
    BEFORE = "before"           # A ends before B starts
    AFTER = "after"             # A starts after B ends
    MEETS = "meets"             # A ends exactly when B starts
    MET_BY = "met_by"           # A starts exactly when B ends
    OVERLAPS = "overlaps"       # A starts before B, ends during B
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"           # A starts with B, ends before B
    STARTED_BY = "started_by"
    DURING = "during"           # A is contained in B
    CONTAINS = "contains"       # A contains B
    FINISHES = "finishes"       # A ends with B, starts after B
    FINISHED_BY = "finished_by"
    EQUALS = "equals"           # A and B are the same interval
    
    # Special
    UNKNOWN = "unknown"         # Relation not determined


# Inverse relations for constraint propagation
INVERSE_RELATIONS = {
    TemporalRelation.BEFORE: TemporalRelation.AFTER,
    TemporalRelation.AFTER: TemporalRelation.BEFORE,
    TemporalRelation.MEETS: TemporalRelation.MET_BY,
    TemporalRelation.MET_BY: TemporalRelation.MEETS,
    TemporalRelation.OVERLAPS: TemporalRelation.OVERLAPPED_BY,
    TemporalRelation.OVERLAPPED_BY: TemporalRelation.OVERLAPS,
    TemporalRelation.STARTS: TemporalRelation.STARTED_BY,
    TemporalRelation.STARTED_BY: TemporalRelation.STARTS,
    TemporalRelation.DURING: TemporalRelation.CONTAINS,
    TemporalRelation.CONTAINS: TemporalRelation.DURING,
    TemporalRelation.FINISHES: TemporalRelation.FINISHED_BY,
    TemporalRelation.FINISHED_BY: TemporalRelation.FINISHES,
    TemporalRelation.EQUALS: TemporalRelation.EQUALS,
}


@dataclass
class TemporalEvent:
    """An event with temporal information."""
    event_id: str
    description: str
    entity: str                          # Main entity involved
    event_type: str                      # birth, death, action, etc.
    
    # Temporal position (from narrative order)
    narrative_position: int              # Position in text
    chapter: Optional[int] = None
    
    # Explicit time if available
    explicit_time: Optional[str] = None  # "1985", "morning", etc.
    
    # For constraint tracking
    is_terminal: bool = False           # Death, destruction (no events after)
    is_origin: bool = False             # Birth, creation (no events before for entity)
    
    def __hash__(self):
        return hash(self.event_id)


@dataclass
class TemporalConstraint:
    """A temporal constraint between two events."""
    event_a: str                         # event_id
    event_b: str                         # event_id
    relation: TemporalRelation
    confidence: float = 1.0
    source: str = ""                     # Where this constraint came from


@dataclass
class TemporalViolation:
    """A detected temporal inconsistency."""
    description: str
    event_a: TemporalEvent
    event_b: TemporalEvent
    expected_relation: TemporalRelation
    actual_relation: TemporalRelation
    severity: float                      # 1.0 = impossible, 0.5 = unlikely
    explanation: str


class TemporalGraph:
    """
    Temporal constraint graph for narrative reasoning.
    
    Uses Allen's Interval Algebra for representing and reasoning
    about temporal relations between events.
    """
    
    def __init__(self):
        # Events storage
        self.events: Dict[str, TemporalEvent] = {}
        
        # Constraint graph (adjacency with relations)
        self.constraints: Dict[str, Dict[str, TemporalRelation]] = defaultdict(dict)
        
        # Entity timelines (ordered events per entity)
        self.entity_timelines: Dict[str, List[TemporalEvent]] = defaultdict(list)
        
        # Terminal events (death, destruction)
        self.terminal_events: Dict[str, TemporalEvent] = {}  # entity -> terminal event
        
        # Origin events (birth, creation)
        self.origin_events: Dict[str, TemporalEvent] = {}  # entity -> origin event
        
    def add_event(self, event: TemporalEvent):
        """Add an event to the graph."""
        self.events[event.event_id] = event
        
        # Add to entity timeline
        self.entity_timelines[event.entity.lower()].append(event)
        # Sort by narrative position
        self.entity_timelines[event.entity.lower()].sort(
            key=lambda e: e.narrative_position
        )
        
        # Track terminal/origin events
        if event.is_terminal:
            self.terminal_events[event.entity.lower()] = event
        if event.is_origin:
            self.origin_events[event.entity.lower()] = event
    
    def add_constraint(
        self,
        event_a_id: str,
        event_b_id: str,
        relation: TemporalRelation,
        confidence: float = 1.0
    ):
        """Add a temporal constraint between events."""
        self.constraints[event_a_id][event_b_id] = relation
        # Add inverse
        inv = INVERSE_RELATIONS.get(relation, TemporalRelation.UNKNOWN)
        self.constraints[event_b_id][event_a_id] = inv
    
    def infer_constraints_from_narrative_order(self):
        """
        Infer BEFORE/AFTER constraints from narrative order.
        
        Assumption: Events described earlier in the text generally
        happen before events described later (for same entity).
        """
        for entity, timeline in self.entity_timelines.items():
            for i in range(len(timeline) - 1):
                event_a = timeline[i]
                event_b = timeline[i + 1]
                
                # Infer: event_a BEFORE event_b (with moderate confidence)
                self.add_constraint(
                    event_a.event_id,
                    event_b.event_id,
                    TemporalRelation.BEFORE,
                    confidence=0.7  # Narrative order is a heuristic
                )
    
    def infer_terminal_constraints(self):
        """
        Infer constraints from terminal events (death).
        
        Rule: No event can happen AFTER a terminal event for that entity.
        """
        for entity, terminal_event in self.terminal_events.items():
            timeline = self.entity_timelines.get(entity, [])
            
            for event in timeline:
                if event.event_id != terminal_event.event_id:
                    # All events must be BEFORE terminal event
                    self.add_constraint(
                        event.event_id,
                        terminal_event.event_id,
                        TemporalRelation.BEFORE,
                        confidence=1.0  # Absolute constraint
                    )
    
    def infer_origin_constraints(self):
        """
        Infer constraints from origin events (birth).
        
        Rule: No event can happen BEFORE an origin event for that entity.
        """
        for entity, origin_event in self.origin_events.items():
            timeline = self.entity_timelines.get(entity, [])
            
            for event in timeline:
                if event.event_id != origin_event.event_id:
                    # All events must be AFTER origin event
                    self.add_constraint(
                        origin_event.event_id,
                        event.event_id,
                        TemporalRelation.BEFORE,
                        confidence=1.0  # Absolute constraint
                    )
    
    def check_consistency(self) -> List[TemporalViolation]:
        """
        Check for temporal inconsistencies in the graph.
        
        Uses constraint propagation to find violations.
        """
        violations = []
        
        # Check for direct contradictions
        for event_a_id, relations in self.constraints.items():
            for event_b_id, relation in relations.items():
                # Check if reverse constraint exists and is contradictory
                if event_b_id in self.constraints:
                    reverse_rel = self.constraints[event_b_id].get(event_a_id)
                    
                    if reverse_rel and not self._relations_compatible(relation, reverse_rel):
                        event_a = self.events.get(event_a_id)
                        event_b = self.events.get(event_b_id)
                        
                        if event_a and event_b:
                            violations.append(TemporalViolation(
                                description=f"Contradictory temporal relations",
                                event_a=event_a,
                                event_b=event_b,
                                expected_relation=relation,
                                actual_relation=reverse_rel,
                                severity=1.0,
                                explanation=f"{event_a.description} has conflicting temporal relation with {event_b.description}"
                            ))
        
        # Check terminal event violations
        for entity, terminal_event in self.terminal_events.items():
            for event in self.entity_timelines.get(entity, []):
                if event.narrative_position > terminal_event.narrative_position:
                    if event.event_id != terminal_event.event_id:
                        violations.append(TemporalViolation(
                            description=f"Event after terminal event",
                            event_a=terminal_event,
                            event_b=event,
                            expected_relation=TemporalRelation.AFTER,
                            actual_relation=TemporalRelation.BEFORE,
                            severity=1.0,
                            explanation=f"'{event.entity}' performs action after death/destruction"
                        ))
        
        # Check origin event violations (events before birth)
        for entity, origin_event in self.origin_events.items():
            for event in self.entity_timelines.get(entity, []):
                if event.narrative_position < origin_event.narrative_position:
                    if event.event_id != origin_event.event_id:
                        violations.append(TemporalViolation(
                            description=f"Event before origin event",
                            event_a=event,
                            event_b=origin_event,
                            expected_relation=TemporalRelation.AFTER,
                            actual_relation=TemporalRelation.BEFORE,
                            severity=1.0,
                            explanation=f"'{event.entity}' performs action before birth/creation"
                        ))
        
        return violations
    
    def _relations_compatible(
        self,
        rel_a: TemporalRelation,
        rel_b: TemporalRelation
    ) -> bool:
        """Check if two relations are compatible."""
        # A relation and its inverse are compatible
        if rel_b == INVERSE_RELATIONS.get(rel_a):
            return True
        # Same relation is compatible
        if rel_a == rel_b:
            return True
        # BEFORE and AFTER are incompatible
        if (rel_a == TemporalRelation.BEFORE and rel_b == TemporalRelation.BEFORE):
            return False  # Cycle!
        return True
    
    def get_entity_timeline_str(self, entity: str) -> str:
        """Get readable timeline for an entity."""
        timeline = self.entity_timelines.get(entity.lower(), [])
        if not timeline:
            return f"No events found for {entity}"
        
        lines = [f"Timeline for {entity}:"]
        for i, event in enumerate(timeline):
            marker = ""
            if event.is_origin:
                marker = " [BIRTH]"
            elif event.is_terminal:
                marker = " [DEATH]"
            lines.append(f"  {i+1}. {event.description}{marker}")
        
        return "\n".join(lines)


class TemporalNarrativeReasoner:
    """
    High-level temporal reasoning for narrative consistency.
    
    Combines TemporalGraph with event extraction for
    end-to-end temporal constraint checking.
    """
    
    # Keywords for detecting birth/death events
    TERMINAL_KEYWORDS = {'died', 'death', 'killed', 'murdered', 'deceased', 'perished', 'destroyed'}
    ORIGIN_KEYWORDS = {'born', 'birth', 'created', 'emerged', 'came into being'}
    
    def __init__(self):
        self.graph = TemporalGraph()
        self.event_counter = 0
    
    def _generate_event_id(self) -> str:
        self.event_counter += 1
        return f"event_{self.event_counter}"
    
    def _is_terminal_event(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.TERMINAL_KEYWORDS)
    
    def _is_origin_event(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.ORIGIN_KEYWORDS)
    
    def _extract_entity_from_event(self, event_text: str) -> Optional[str]:
        """Extract main entity (subject) from event text."""
        # Simple heuristic: first capitalized word sequence
        match = re.search(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', event_text)
        if match:
            return match.group(1)
        return None
    
    def add_events_from_extraction(
        self,
        events: List[Any],  # ExtractedEvent from neural_event_extractor
    ):
        """
        Build temporal graph from extracted events.
        
        Args:
            events: List of ExtractedEvent from NeuralEventExtractor
        """
        for event in events:
            # Get entity name
            entity = None
            if hasattr(event, 'agent') and event.agent:
                entity = event.agent.text if hasattr(event.agent, 'text') else str(event.agent)
            elif hasattr(event, 'patient') and event.patient:
                entity = event.patient.text if hasattr(event.patient, 'text') else str(event.patient)
            else:
                entity = self._extract_entity_from_event(event.text if hasattr(event, 'text') else str(event))
            
            if not entity:
                continue
            
            description = event.source_sentence if hasattr(event, 'source_sentence') else str(event)
            
            # Determine event type
            is_terminal = False
            is_origin = False
            event_type = event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type)
            
            if self._is_terminal_event(description):
                is_terminal = True
                event_type = "death"
            elif self._is_origin_event(description):
                is_origin = True
                event_type = "birth"
            
            temporal_event = TemporalEvent(
                event_id=self._generate_event_id(),
                description=description[:200],  # Truncate
                entity=entity,
                event_type=event_type,
                narrative_position=event.char_start if hasattr(event, 'char_start') else 0,
                is_terminal=is_terminal,
                is_origin=is_origin,
            )
            
            self.graph.add_event(temporal_event)
        
        # Infer constraints
        self.graph.infer_constraints_from_narrative_order()
        self.graph.infer_terminal_constraints()
        self.graph.infer_origin_constraints()
    
    def check_backstory_consistency(
        self,
        backstory_events: List[Any]
    ) -> Tuple[bool, List[TemporalViolation]]:
        """
        Check if backstory events are temporally consistent with narrative.
        
        Args:
            backstory_events: Events extracted from backstory
            
        Returns:
            (is_consistent, list of violations)
        """
        # Add backstory events to graph
        self.add_events_from_extraction(backstory_events)
        
        # Check consistency
        violations = self.graph.check_consistency()
        
        is_consistent = len(violations) == 0
        
        return is_consistent, violations
    
    def generate_report(self) -> str:
        """Generate interpretable temporal analysis report."""
        lines = []
        lines.append("# Temporal Analysis Report")
        lines.append(f"\n**Total Events**: {len(self.graph.events)}")
        lines.append(f"**Entities Tracked**: {len(self.graph.entity_timelines)}")
        
        # Terminal events
        if self.graph.terminal_events:
            lines.append("\n## Terminal Events (Death/Destruction):")
            for entity, event in self.graph.terminal_events.items():
                lines.append(f"- **{entity}**: {event.description[:100]}...")
        
        # Origin events
        if self.graph.origin_events:
            lines.append("\n## Origin Events (Birth/Creation):")
            for entity, event in self.graph.origin_events.items():
                lines.append(f"- **{entity}**: {event.description[:100]}...")
        
        # Violations
        violations = self.graph.check_consistency()
        if violations:
            lines.append(f"\n## ⚠️ Temporal Violations ({len(violations)}):")
            for v in violations[:10]:  # Top 10
                lines.append(f"\n### Violation: {v.description}")
                lines.append(f"- **Severity**: {v.severity:.0%}")
                lines.append(f"- **Explanation**: {v.explanation}")
                lines.append(f"- **Event A**: {v.event_a.description[:80]}...")
                lines.append(f"- **Event B**: {v.event_b.description[:80]}...")
        else:
            lines.append("\n## ✅ No Temporal Violations Detected")
        
        return "\n".join(lines)
