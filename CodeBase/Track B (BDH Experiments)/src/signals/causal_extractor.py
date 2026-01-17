"""
Causal Event Extraction and Constraint Tracking

Addresses the blueprint's requirement for:
1. Tracking "state changes" (character dying, items destroyed, relationships altered)
2. Building a constraint graph (X cannot happen if Y)
3. Detecting violations of narrative constraints

This implements EXPLICIT causal reasoning, not just implicit perplexity signals.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum
import re


class EventType(Enum):
    """Types of narrative events that create constraints."""
    STATE_CHANGE = "state_change"       # Character/object state changes
    RELATIONSHIP = "relationship"        # Relationships form/break
    LOCATION = "location"                # Character moves to location
    POSSESSION = "possession"            # Character gains/loses item
    TEMPORAL = "temporal"                # Time-based event
    EXISTENCE = "existence"              # Birth, death, creation, destruction
    ABILITY = "ability"                  # Character gains/loses ability
    KNOWLEDGE = "knowledge"              # Character learns/forgets info


class ConstraintType(Enum):
    """Types of logical constraints."""
    MUTEX = "mutex"                      # X and Y cannot both be true
    REQUIRES = "requires"                # X requires Y to be true
    TEMPORAL = "temporal"                # X must happen before Y
    NEGATION = "negation"                # X means NOT Y


@dataclass
class NarrativeEvent:
    """An event extracted from the narrative."""
    text: str
    event_type: EventType
    subject: str                         # Who/what is involved
    predicate: str                       # What happened
    object: Optional[str] = None         # Target of action (if any)
    location: Optional[str] = None       # Where it happened
    chapter: int = 0                     # Approximate position
    token_position: int = 0              # Exact position
    confidence: float = 1.0
    embedding: Optional[torch.Tensor] = field(default=None, repr=False)


@dataclass
class Constraint:
    """A logical constraint derived from events."""
    constraint_type: ConstraintType
    source_event: NarrativeEvent
    constraint_text: str                 # Human-readable constraint
    subjects: Set[str]                   # Entities affected
    valid_from: int = 0                  # Token position where constraint starts
    valid_until: int = float('inf')     # Token position where constraint ends
    

@dataclass
class ConstraintViolation:
    """A detected violation of a narrative constraint."""
    constraint: Constraint
    violating_claim: str
    explanation: str
    severity: float                      # 0-1, how severe the violation is


class CausalEventExtractor:
    """
    Extracts events from narrative text and builds constraint relationships.
    
    This enables EXPLICIT causal tracking rather than implicit perplexity.
    """
    
    # Event detection patterns
    STATE_CHANGE_PATTERNS = [
        r"(\w+)\s+(became|turned into|transformed into|changed into)\s+(.+)",
        r"(\w+)\s+(was|is)\s+now\s+(.+)",
        r"(\w+)\s+(died|deceased|passed away|was killed|perished)",
        r"(\w+)\s+(was born|came into being|was created)",
    ]
    
    RELATIONSHIP_PATTERNS = [
        r"(\w+)\s+(married|wed|was engaged to)\s+(\w+)",
        r"(\w+)\s+(is|was)\s+(the|a)\s+(father|mother|brother|sister|friend|enemy)\s+of\s+(\w+)",
        r"(\w+)\s+and\s+(\w+)\s+(became friends|broke up|reconciled)",
    ]
    
    LOCATION_PATTERNS = [
        r"(\w+)\s+(went to|traveled to|arrived at|was in|lived in)\s+(.+)",
        r"(\w+)\s+(left|departed from|fled)\s+(.+)",
        r"In\s+(.+),\s+(\w+)\s+",
    ]
    
    POSSESSION_PATTERNS = [
        r"(\w+)\s+(found|obtained|received|was given|stole|took)\s+(?:the\s+)?(.+)",
        r"(\w+)\s+(lost|gave away|destroyed|sold)\s+(?:the\s+)?(.+)",
        r"(?:the\s+)?(.+)\s+(?:belonged to|was owned by)\s+(\w+)",
    ]
    
    EXISTENCE_PATTERNS = [
        r"(\w+)\s+(was born|came into existence|was created)",
        r"(\w+)\s+(died|was destroyed|ceased to exist|perished)",
        r"(\w+)\s+was\s+(alive|dead)\s+",
    ]
    
    def __init__(self, model: Optional[nn.Module] = None, tokenizer=None):
        """
        Args:
            model: Optional model for embedding events
            tokenizer: Optional tokenizer for text processing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.events: List[NarrativeEvent] = []
        self.constraints: List[Constraint] = []
        
    def extract_events(
        self,
        text: str,
        use_patterns: bool = True,
        use_model: bool = False
    ) -> List[NarrativeEvent]:
        """
        Extract narrative events from text.
        
        Args:
            text: Narrative text
            use_patterns: Use regex patterns (fast, baseline)
            use_model: Use model for extraction (more accurate, slower)
            
        Returns:
            List of extracted events
        """
        events = []
        
        # Split into sentences for processing
        sentences = self._split_sentences(text)
        
        current_position = 0
        for sent_idx, sentence in enumerate(sentences):
            if use_patterns:
                sent_events = self._extract_with_patterns(
                    sentence, 
                    token_position=current_position
                )
                events.extend(sent_events)
            
            if use_model and self.model is not None:
                # Model-based extraction (more sophisticated)
                model_events = self._extract_with_model(
                    sentence,
                    token_position=current_position
                )
                # Merge with pattern-based events
                events = self._merge_events(events, model_events)
            
            current_position += len(sentence.split())
        
        self.events = events
        return events
    
    def _extract_with_patterns(
        self,
        sentence: str,
        token_position: int
    ) -> List[NarrativeEvent]:
        """Extract events using regex patterns."""
        events = []
        
        # Check each pattern type
        for pattern in self.STATE_CHANGE_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                events.append(NarrativeEvent(
                    text=match.group(0),
                    event_type=EventType.STATE_CHANGE,
                    subject=match.group(1),
                    predicate=match.group(2) if len(match.groups()) > 1 else "",
                    object=match.group(3) if len(match.groups()) > 2 else None,
                    token_position=token_position,
                    confidence=0.8
                ))
        
        for pattern in self.RELATIONSHIP_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                events.append(NarrativeEvent(
                    text=match.group(0),
                    event_type=EventType.RELATIONSHIP,
                    subject=match.group(1),
                    predicate=match.group(2) if len(match.groups()) > 1 else "",
                    object=match.groups()[-1] if len(match.groups()) > 2 else None,
                    token_position=token_position,
                    confidence=0.8
                ))
        
        for pattern in self.LOCATION_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                events.append(NarrativeEvent(
                    text=match.group(0),
                    event_type=EventType.LOCATION,
                    subject=match.group(1) if not match.group(1).startswith('In') else match.group(2),
                    predicate=match.group(2) if len(match.groups()) > 1 else "",
                    location=match.group(3) if len(match.groups()) > 2 else match.group(1),
                    token_position=token_position,
                    confidence=0.7
                ))
        
        for pattern in self.POSSESSION_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                events.append(NarrativeEvent(
                    text=match.group(0),
                    event_type=EventType.POSSESSION,
                    subject=match.group(1),
                    predicate=match.group(2) if len(match.groups()) > 1 else "",
                    object=match.group(3) if len(match.groups()) > 2 else None,
                    token_position=token_position,
                    confidence=0.7
                ))
        
        for pattern in self.EXISTENCE_PATTERNS:
            for match in re.finditer(pattern, sentence, re.IGNORECASE):
                events.append(NarrativeEvent(
                    text=match.group(0),
                    event_type=EventType.EXISTENCE,
                    subject=match.group(1),
                    predicate=match.group(2) if len(match.groups()) > 1 else "",
                    token_position=token_position,
                    confidence=0.9  # Existence events are usually unambiguous
                ))
        
        return events
    
    def _extract_with_model(
        self,
        sentence: str,
        token_position: int
    ) -> List[NarrativeEvent]:
        """
        SOTA: Extract events using BDH hidden states.
        
        Uses attention patterns and learned probes to identify:
        - Subject (WHO)
        - Predicate (WHAT happened)
        - Object (TO WHOM/WHAT)
        - Location (WHERE)
        
        This is more accurate than regex for:
        - Implicit events
        - Complex sentence structures
        - Coreference resolution
        """
        if self.model is None or self.tokenizer is None:
            return []
        
        events = []
        device = next(self.model.parameters()).device
        
        # Tokenize sentence
        encoded = self.tokenizer(
            sentence,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )
        
        input_ids = encoded['input_ids'].to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=True if hasattr(self.model.config, 'output_attentions') else False
            )
            
            hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            
            # Get attention if available (for subject-object relations)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attentions = outputs.attentions[-1]  # Last layer attention
            else:
                attentions = None
        
        # Decode tokens for analysis
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Use hidden state patterns to identify event components
        # This is a simplified SRL approach using activation patterns
        
        # 1. Find potential PREDICATES (verbs) using activation spikes
        # Verbs typically have distinct activation patterns
        hidden_norms = hidden_states[0].norm(dim=-1)  # [seq_len]
        mean_norm = hidden_norms.mean()
        std_norm = hidden_norms.std()
        
        # Tokens with high activation are often semantically important
        important_positions = (hidden_norms > mean_norm + 0.5 * std_norm).nonzero().squeeze(-1)
        
        # 2. For each important token, check if it's an event-related word
        event_keywords = {
            'died': EventType.EXISTENCE,
            'born': EventType.EXISTENCE,
            'killed': EventType.EXISTENCE,
            'married': EventType.RELATIONSHIP,
            'divorced': EventType.RELATIONSHIP,
            'went': EventType.LOCATION,
            'arrived': EventType.LOCATION,
            'left': EventType.LOCATION,
            'traveled': EventType.LOCATION,
            'became': EventType.STATE_CHANGE,
            'turned': EventType.STATE_CHANGE,
            'found': EventType.POSSESSION,
            'lost': EventType.POSSESSION,
            'received': EventType.POSSESSION,
            'gave': EventType.POSSESSION,
            'learned': EventType.KNOWLEDGE,
            'forgot': EventType.KNOWLEDGE,
            'discovered': EventType.KNOWLEDGE,
        }
        
        for pos in important_positions.tolist():
            if pos >= len(tokens):
                continue
            
            token = tokens[pos].lower().replace('##', '').replace('Ġ', '')
            
            # Check if token matches event keyword
            matched_type = None
            for keyword, etype in event_keywords.items():
                if keyword in token or token in keyword:
                    matched_type = etype
                    break
            
            if matched_type is None:
                continue
            
            # 3. Find SUBJECT (likely before predicate) using attention
            subject = self._find_subject(tokens, pos, hidden_states[0], attentions)
            
            # 4. Find OBJECT (likely after predicate)
            obj = self._find_object(tokens, pos, hidden_states[0], attentions)
            
            # 5. Find LOCATION if applicable
            location = self._find_location(tokens, sentence)
            
            # Create event
            event = NarrativeEvent(
                text=sentence,
                event_type=matched_type,
                subject=subject or "Unknown",
                predicate=token,
                object=obj,
                location=location,
                token_position=token_position,
                confidence=0.85,  # Model-based is more confident than regex
                embedding=hidden_states[0, pos].cpu().clone()
            )
            events.append(event)
        
        return events
    
    def _find_subject(
        self, 
        tokens: List[str], 
        predicate_pos: int,
        hidden_states: torch.Tensor,
        attentions: Optional[torch.Tensor]
    ) -> Optional[str]:
        """Find subject using attention patterns or position heuristics."""
        # Look for capitalized word before predicate (likely proper noun/subject)
        for i in range(predicate_pos - 1, max(0, predicate_pos - 10), -1):
            token = tokens[i].replace('##', '').replace('Ġ', '')
            # Check if it looks like a name (capitalized, not special token)
            if token and token[0].isupper() and not token.startswith('['):
                return token
        
        # Fallback: use attention if available
        if attentions is not None and predicate_pos < attentions.shape[-1]:
            # Average attention heads
            avg_attn = attentions[0].mean(dim=0)  # [seq_len, seq_len]
            # Find which token the predicate attends to most (before it)
            attn_to_pred = avg_attn[predicate_pos, :predicate_pos]
            if len(attn_to_pred) > 0:
                most_attended = attn_to_pred.argmax().item()
                return tokens[most_attended].replace('##', '').replace('Ġ', '')
        
        return None
    
    def _find_object(
        self,
        tokens: List[str],
        predicate_pos: int,
        hidden_states: torch.Tensor,
        attentions: Optional[torch.Tensor]
    ) -> Optional[str]:
        """Find object using position after predicate."""
        # Look for capitalized word after predicate
        for i in range(predicate_pos + 1, min(len(tokens), predicate_pos + 10)):
            token = tokens[i].replace('##', '').replace('Ġ', '')
            if token and token[0].isupper() and not token.startswith('['):
                return token
        return None
    
    def _find_location(self, tokens: List[str], sentence: str) -> Optional[str]:
        """Extract location from sentence using preposition patterns."""
        import re
        # Location usually follows "in", "at", "to"
        loc_match = re.search(r'(?:in|at|to)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)', sentence)
        if loc_match:
            return loc_match.group(1)
        return None
    
    def _merge_events(
        self,
        events1: List[NarrativeEvent],
        events2: List[NarrativeEvent]
    ) -> List[NarrativeEvent]:
        """Merge two event lists, deduplicating."""
        # Simple dedup based on text
        seen = set(e.text for e in events1)
        merged = list(events1)
        for e in events2:
            if e.text not in seen:
                merged.append(e)
                seen.add(e.text)
        return merged
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def build_constraints(self) -> List[Constraint]:
        """
        Build logical constraints from extracted events.
        
        This is where we create the "X cannot happen if Y" rules.
        """
        constraints = []
        
        for event in self.events:
            # EXISTENCE constraints
            if event.event_type == EventType.EXISTENCE:
                if "died" in event.predicate.lower() or "dead" in event.predicate.lower():
                    # Death creates MUTEX constraint: cannot act after death
                    constraints.append(Constraint(
                        constraint_type=ConstraintType.MUTEX,
                        source_event=event,
                        constraint_text=f"{event.subject} cannot perform actions after death",
                        subjects={event.subject},
                        valid_from=event.token_position
                    ))
                elif "born" in event.predicate.lower():
                    # Birth creates temporal constraint: cannot act before birth
                    constraints.append(Constraint(
                        constraint_type=ConstraintType.TEMPORAL,
                        source_event=event,
                        constraint_text=f"{event.subject} cannot exist before birth",
                        subjects={event.subject},
                        valid_until=event.token_position
                    ))
            
            # LOCATION constraints
            elif event.event_type == EventType.LOCATION:
                if event.location:
                    # Being in one place means not in another (MUTEX)
                    constraints.append(Constraint(
                        constraint_type=ConstraintType.MUTEX,
                        source_event=event,
                        constraint_text=f"{event.subject} cannot be elsewhere while in {event.location}",
                        subjects={event.subject},
                        valid_from=event.token_position
                    ))
            
            # RELATIONSHIP constraints
            elif event.event_type == EventType.RELATIONSHIP:
                if event.object:
                    # Relationships create logical implications
                    if "brother" in event.predicate or "sister" in event.predicate:
                        constraints.append(Constraint(
                            constraint_type=ConstraintType.REQUIRES,
                            source_event=event,
                            constraint_text=f"{event.subject} and {event.object} share parents",
                            subjects={event.subject, event.object}
                        ))
                    if "enemy" in event.predicate:
                        constraints.append(Constraint(
                            constraint_type=ConstraintType.NEGATION,
                            source_event=event,
                            constraint_text=f"{event.subject} and {event.object} are not friends",
                            subjects={event.subject, event.object}
                        ))
        
        self.constraints = constraints
        return constraints


class ConstraintViolationDetector:
    """
    Detects violations of narrative constraints in backstory claims.
    """
    
    def __init__(
        self,
        event_extractor: CausalEventExtractor,
        model: Optional[nn.Module] = None,
        tokenizer = None
    ):
        self.event_extractor = event_extractor
        self.model = model
        self.tokenizer = tokenizer
    
    def check_backstory(
        self,
        backstory_text: str
    ) -> List[ConstraintViolation]:
        """
        Check if backstory violates any established constraints.
        
        Args:
            backstory_text: Hypothetical backstory
            
        Returns:
            List of detected violations
        """
        # Extract events from backstory
        backstory_events = self.event_extractor.extract_events(backstory_text)
        
        violations = []
        
        for bs_event in backstory_events:
            for constraint in self.event_extractor.constraints:
                # Check if backstory event involves same subjects
                if bs_event.subject not in constraint.subjects:
                    continue
                
                violation = self._check_violation(bs_event, constraint)
                if violation:
                    violations.append(violation)
        
        return violations
    
    def _check_violation(
        self,
        event: NarrativeEvent,
        constraint: Constraint
    ) -> Optional[ConstraintViolation]:
        """Check if a single event violates a constraint."""
        
        if constraint.constraint_type == ConstraintType.MUTEX:
            # Check if event contradicts mutex constraint
            if "death" in constraint.constraint_text.lower():
                # Any action by dead character is violation
                return ConstraintViolation(
                    constraint=constraint,
                    violating_claim=event.text,
                    explanation=f"'{event.subject}' performs action after established death",
                    severity=1.0
                )
        
        elif constraint.constraint_type == ConstraintType.TEMPORAL:
            # Check temporal ordering
            if event.token_position < constraint.valid_from:
                return ConstraintViolation(
                    constraint=constraint,
                    violating_claim=event.text,
                    explanation=f"'{event.subject}' acts before established birth/existence",
                    severity=0.9
                )
        
        elif constraint.constraint_type == ConstraintType.NEGATION:
            # Check negation
            source_pred = constraint.source_event.predicate.lower()
            event_pred = event.predicate.lower()
            
            # Simple contradiction check
            if ("enemy" in source_pred and "friend" in event_pred) or \
               ("friend" in source_pred and "enemy" in event_pred):
                return ConstraintViolation(
                    constraint=constraint,
                    violating_claim=event.text,
                    explanation=f"Relationship contradiction: {constraint.constraint_text}",
                    severity=0.8
                )
        
        return None
    
    def get_constraint_summary(self) -> Dict[str, int]:
        """Get summary of constraints by type."""
        summary = {}
        for c in self.event_extractor.constraints:
            ctype = c.constraint_type.value
            summary[ctype] = summary.get(ctype, 0) + 1
        return summary
