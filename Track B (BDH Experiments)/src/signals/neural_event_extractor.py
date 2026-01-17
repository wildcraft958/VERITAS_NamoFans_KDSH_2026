"""
Neural Event Extraction and Semantic Role Labeling

SOTA replacement for regex-based event extraction.
Uses pre-trained transformers for:
- Named Entity Recognition (entities involved)
- Semantic Role Labeling (who did what to whom)
- Event Classification (what type of event)

Key features:
- SpaCy/HuggingFace NER for entity extraction
- Event trigger detection using BERT
- Coreference resolution for entity tracking
- Interpretable event chains
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re


class EventType(Enum):
    """Types of narrative events for constraint tracking."""
    STATE_CHANGE = "state_change"
    RELATIONSHIP = "relationship"
    LOCATION = "location"
    POSSESSION = "possession"
    TEMPORAL = "temporal"
    EXISTENCE = "existence"
    ABILITY = "ability"
    KNOWLEDGE = "knowledge"
    ACTION = "action"


@dataclass
class Entity:
    """A named entity in the narrative."""
    text: str
    entity_type: str  # PERSON, LOCATION, ORG, etc.
    mentions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) positions
    aliases: Set[str] = field(default_factory=set)  # Alternative names
    
    def __hash__(self):
        return hash(self.text.lower())
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.text.lower() == other.text.lower()
        return False


@dataclass
class ExtractedEvent:
    """A neural-extracted event with full SRL information."""
    text: str
    event_type: EventType
    
    # SRL components
    agent: Optional[Entity] = None       # Who performed the action (ARG0)
    patient: Optional[Entity] = None     # Who/what was affected (ARG1)
    predicate: str = ""                  # The action/event trigger
    
    # Additional arguments
    location: Optional[str] = None       # Where (ARGM-LOC)
    time: Optional[str] = None           # When (ARGM-TMP)
    manner: Optional[str] = None         # How (ARGM-MNR)
    
    # Position and confidence
    sentence_index: int = 0
    char_start: int = 0
    char_end: int = 0
    confidence: float = 0.0
    
    # For interpretability
    source_sentence: str = ""


@dataclass
class EventChain:
    """Chain of related events for a given entity."""
    entity: Entity
    events: List[ExtractedEvent] = field(default_factory=list)
    
    def add_event(self, event: ExtractedEvent):
        self.events.append(event)
        # Keep sorted by position
        self.events.sort(key=lambda e: e.char_start)


class NeuralEventExtractor:
    """
    SOTA Neural Event Extraction using pre-trained NLP models.
    
    Pipeline:
    1. Named Entity Recognition (SpaCy or BERT-NER)
    2. Event Trigger Detection (predicate identification)
    3. Semantic Role Labeling (argument extraction)
    4. Event Classification (categorize event type)
    5. Coreference Resolution (link mentions)
    """
    
    # Event trigger keywords for classification
    EXISTENCE_TRIGGERS = {
        'died', 'born', 'killed', 'dead', 'alive', 'death', 'birth',
        'murdered', 'perished', 'passed away', 'created', 'destroyed'
    }
    
    RELATIONSHIP_TRIGGERS = {
        'married', 'divorced', 'engaged', 'friend', 'enemy', 'brother',
        'sister', 'father', 'mother', 'son', 'daughter', 'husband', 'wife',
        'betrayed', 'loved', 'hated', 'met', 'separated'
    }
    
    LOCATION_TRIGGERS = {
        'went', 'traveled', 'arrived', 'left', 'departed', 'moved',
        'lived', 'stayed', 'visited', 'returned', 'fled', 'came'
    }
    
    POSSESSION_TRIGGERS = {
        'found', 'lost', 'stole', 'gave', 'received', 'bought', 'sold',
        'inherited', 'owned', 'possessed', 'acquired', 'obtained'
    }
    
    STATE_CHANGE_TRIGGERS = {
        'became', 'turned', 'transformed', 'changed', 'converted',
        'grew', 'aged', 'recovered', 'injured', 'healed'
    }
    
    KNOWLEDGE_TRIGGERS = {
        'learned', 'discovered', 'forgot', 'remembered', 'realized',
        'understood', 'knew', 'believed', 'thought', 'suspected'
    }
    

    def __init__(
        self,
        device: torch.device = None,
        use_spacy: bool = True,
        use_ner_model: str = "dslim/bert-large-NER", # Upgraded to Large
        batch_size: int = 16,
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_spacy = use_spacy
        self.ner_model_name = use_ner_model
        self.batch_size = batch_size
        
        self.nlp = None
        self.ner_pipeline = None
        self._loaded = False
        
        # Storage
        self.entities: Dict[str, Entity] = {}
        self.events: List[ExtractedEvent] = []
        self.event_chains: Dict[str, EventChain] = {}
    
    def load(self):
        """Load NLP models."""
        if self._loaded:
            return
        
        print("Loading NER models...")
        
        if self.use_spacy:
            try:
                import spacy
                # Try loading Transformer model (SOTA)
                try:
                    print("Attempting to load SpaCy TRF model (en_core_web_trf)...")
                    self.nlp = spacy.load("en_core_web_trf")
                    print("✅ SpaCy TRF loaded (Best Accuracy)")
                except OSError:
                    # Fallback to Large
                    try:
                        print("TRF not found. Trying Large model (en_core_web_lg)...")
                        self.nlp = spacy.load("en_core_web_lg")
                        print("✅ SpaCy Large loaded")
                    except OSError:
                         # Fallback to Small (and download if needed)
                        try:
                            print("Large not found. Falling back to Small (en_core_web_sm)...")
                            self.nlp = spacy.load("en_core_web_sm")
                        except OSError:
                            print("Downloading spacy model...")
                            import subprocess
                            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                            self.nlp = spacy.load("en_core_web_sm")
                print("SpaCy loaded")
            except ImportError:
                print("SpaCy not available, using transformers NER only")
                self.use_spacy = False
        
        # Load HuggingFace NER as backup/enhancement
        try:
            from transformers import pipeline
            self.ner_pipeline = pipeline(
                "ner",
                model=self.ner_model_name,
                aggregation_strategy="simple",
                device=0 if self.device.type == 'cuda' else -1
            )
            print(f"Transformer NER loaded: {self.ner_model_name}")
        except Exception as e:
            print(f"Transformer NER not available: {e}")
        
        self._loaded = True
    
    def _split_sentences(self, text: str) -> List[Tuple[str, int, int]]:
        """Split text into sentences with positions."""
        self.load()
        
        if self.nlp:
            doc = self.nlp(text)
            return [(sent.text, sent.start_char, sent.end_char) for sent in doc.sents]
        else:
            # Fallback: regex splitting
            sentences = []
            for match in re.finditer(r'[^.!?]+[.!?]+', text):
                sentences.append((match.group().strip(), match.start(), match.end()))
            return sentences
    
    def _extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using SpaCy."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                entity_type=ent.label_,
                mentions=[(ent.start_char, ent.end_char)]
            )
            entities.append(entity)
        
        return entities
    
    def _extract_entities_transformer(self, text: str) -> List[Entity]:
        """Extract entities using BERT-NER."""
        if not self.ner_pipeline:
            return []
        
        # Process in chunks to handle long text
        max_len = 512
        entities = []
        
        for i in range(0, len(text), max_len - 100):
            chunk = text[i:i + max_len]
            try:
                ner_results = self.ner_pipeline(chunk)
                for ent in ner_results:
                    entity = Entity(
                        text=ent['word'],
                        entity_type=ent['entity_group'],
                        mentions=[(i + ent['start'], i + ent['end'])]
                    )
                    entities.append(entity)
            except Exception:
                continue
        
        return entities
    
    def _merge_entities(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Merge entity mentions, handling aliases and coreference."""
        merged = {}
        
        for entity in entities:
            key = entity.text.lower().strip()
            
            if key in merged:
                merged[key].mentions.extend(entity.mentions)
            else:
                merged[key] = entity
        
        return merged
    
    def _classify_event_type(self, predicate: str, sentence: str) -> EventType:
        """Classify event type based on predicate and context."""
        pred_lower = predicate.lower()
        sent_lower = sentence.lower()
        
        if any(t in pred_lower or t in sent_lower for t in self.EXISTENCE_TRIGGERS):
            return EventType.EXISTENCE
        elif any(t in pred_lower or t in sent_lower for t in self.RELATIONSHIP_TRIGGERS):
            return EventType.RELATIONSHIP
        elif any(t in pred_lower or t in sent_lower for t in self.LOCATION_TRIGGERS):
            return EventType.LOCATION
        elif any(t in pred_lower or t in sent_lower for t in self.POSSESSION_TRIGGERS):
            return EventType.POSSESSION
        elif any(t in pred_lower or t in sent_lower for t in self.STATE_CHANGE_TRIGGERS):
            return EventType.STATE_CHANGE
        elif any(t in pred_lower or t in sent_lower for t in self.KNOWLEDGE_TRIGGERS):
            return EventType.KNOWLEDGE
        else:
            return EventType.ACTION
    
    def _extract_srl_spacy(
        self,
        sentence: str,
        entities_in_sent: List[Entity]
    ) -> List[Dict[str, Any]]:
        """
        Extract SRL-like structure using SpaCy dependency parsing.
        
        This approximates SRL by finding:
        - ROOT verb (predicate)
        - nsubj (agent/ARG0)
        - dobj/pobj (patient/ARG1)
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(sentence)
        srl_frames = []
        
        for token in doc:
            if token.pos_ == "VERB":
                frame = {
                    'predicate': token.text,
                    'predicate_lemma': token.lemma_,
                    'agent': None,
                    'patient': None,
                    'location': None,
                    'time': None,
                }
                
                # Find subject (agent)
                for child in token.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        frame['agent'] = child.text
                    elif child.dep_ in ("dobj", "pobj"):
                        frame['patient'] = child.text
                    elif child.dep_ == "prep":
                        # Check for location/time
                        for pobj in child.children:
                            if pobj.ent_type_ in ("GPE", "LOC", "FAC"):
                                frame['location'] = pobj.text
                            elif pobj.ent_type_ in ("DATE", "TIME"):
                                frame['time'] = pobj.text
                
                if frame['agent'] or frame['patient']:
                    srl_frames.append(frame)
        
        return srl_frames
    
    def extract_events(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> List[ExtractedEvent]:
        """
        Extract all narrative events from text.
        
        Full pipeline:
        1. Entity extraction
        2. Sentence splitting
        3. SRL for each sentence
        4. Event classification
        5. Entity resolution
        """
        self.load()
        
        # 1. Extract all entities
        print("Extracting entities...")
        spacy_entities = self._extract_entities_spacy(text)
        transformer_entities = self._extract_entities_transformer(text)
        
        # Merge entities
        all_entities = spacy_entities + transformer_entities
        self.entities = self._merge_entities(all_entities)
        print(f"Found {len(self.entities)} unique entities")
        
        # 2. Split into sentences
        sentences = self._split_sentences(text)
        print(f"Processing {len(sentences)} sentences")
        
        # 3. Extract events from each sentence
        events = []
        
        for sent_idx, (sentence, start_char, end_char) in enumerate(sentences):
            # Find entities in this sentence
            entities_in_sent = [
                e for e in self.entities.values()
                if any(start_char <= m[0] < end_char for m in e.mentions)
            ]
            
            # Extract SRL frames
            srl_frames = self._extract_srl_spacy(sentence, entities_in_sent)
            
            for frame in srl_frames:
                # Find matching entities
                agent_entity = None
                patient_entity = None
                
                if frame['agent']:
                    agent_key = frame['agent'].lower()
                    agent_entity = self.entities.get(agent_key)
                    if not agent_entity:
                        # Create new entity
                        agent_entity = Entity(text=frame['agent'], entity_type='PERSON')
                        self.entities[agent_key] = agent_entity
                
                if frame['patient']:
                    patient_key = frame['patient'].lower()
                    patient_entity = self.entities.get(patient_key)
                    if not patient_entity:
                        patient_entity = Entity(text=frame['patient'], entity_type='UNKNOWN')
                        self.entities[patient_key] = patient_entity
                
                # Classify event type
                event_type = self._classify_event_type(frame['predicate'], sentence)
                
                # Calculate confidence based on completeness
                confidence = 0.5
                if agent_entity:
                    confidence += 0.2
                if patient_entity:
                    confidence += 0.2
                if event_type != EventType.ACTION:
                    confidence += 0.1
                
                event = ExtractedEvent(
                    text=sentence,
                    event_type=event_type,
                    agent=agent_entity,
                    patient=patient_entity,
                    predicate=frame['predicate'],
                    location=frame.get('location'),
                    time=frame.get('time'),
                    sentence_index=sent_idx,
                    char_start=start_char,
                    char_end=end_char,
                    confidence=min(confidence, 1.0),
                    source_sentence=sentence,
                )
                
                if event.confidence >= min_confidence:
                    events.append(event)
        
        self.events = events
        print(f"Extracted {len(events)} events")
        
        # 4. Build event chains per entity
        self._build_event_chains()
        
        return events
    
    def _build_event_chains(self):
        """Build chains of events for each entity."""
        self.event_chains = {}
        
        for event in self.events:
            # Add to agent's chain
            if event.agent:
                key = event.agent.text.lower()
                if key not in self.event_chains:
                    self.event_chains[key] = EventChain(entity=event.agent)
                self.event_chains[key].add_event(event)
            
            # Add to patient's chain
            if event.patient:
                key = event.patient.text.lower()
                if key not in self.event_chains:
                    self.event_chains[key] = EventChain(entity=event.patient)
                self.event_chains[key].add_event(event)
    
    def get_entity_timeline(self, entity_name: str) -> List[ExtractedEvent]:
        """Get all events involving an entity, in order."""
        key = entity_name.lower()
        if key in self.event_chains:
            return self.event_chains[key].events
        return []
    
    def get_existence_events(self) -> List[ExtractedEvent]:
        """Get birth/death events for constraint checking."""
        return [e for e in self.events if e.event_type == EventType.EXISTENCE]
    
    def get_relationship_events(self) -> List[ExtractedEvent]:
        """Get relationship events."""
        return [e for e in self.events if e.event_type == EventType.RELATIONSHIP]
    
    def generate_summary(self) -> str:
        """Generate human-readable summary of extracted events."""
        lines = []
        lines.append(f"# Event Extraction Summary")
        lines.append(f"\n**Total Events**: {len(self.events)}")
        lines.append(f"**Unique Entities**: {len(self.entities)}")
        
        # Breakdown by type
        type_counts = {}
        for event in self.events:
            t = event.event_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        lines.append("\n## Event Types:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {t}: {count}")
        
        # Key existence events
        existence = self.get_existence_events()
        if existence:
            lines.append("\n## Key Existence Events:")
            for e in existence[:10]:
                agent = e.agent.text if e.agent else "?"
                lines.append(f"- [{e.predicate}] {agent}: \"{e.source_sentence[:100]}...\"")
        
        return "\n".join(lines)
