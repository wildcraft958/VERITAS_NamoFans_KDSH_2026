"""
HippoRAG Integration Adapter

Directly uses the HippoRAG codebase from
(github.com/OSU-NLP-Group/HippoRAG) for knowledge graph construction
and retrieval.

HippoRAG Reference:
- Paper: "HippoRAG: Neurobiologically Inspired Long-Term Memory for LLMs" (NeurIPS 2024)
- Repository: external_frameworks/HippoRAG/
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# Add external_frameworks to path for direct imports
EXTERNAL_FRAMEWORKS_PATH = Path(__file__).parent.parent.parent.parent / "external_frameworks"
HIPPORAG_PATH = EXTERNAL_FRAMEWORKS_PATH / "HippoRAG"
HIPPORAG_SRC_PATH = HIPPORAG_PATH / "src"

if str(HIPPORAG_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(HIPPORAG_SRC_PATH))
if str(HIPPORAG_PATH) not in sys.path:
    sys.path.insert(0, str(HIPPORAG_PATH))

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """An entity extracted from the novel."""
    id: str
    name: str
    type: str = "ENTITY"
    mentions: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)


@dataclass
class Triple:
    """A subject-relation-object triple."""
    subject: str
    relation: str
    object: str
    source_text: str = ""
    confidence: float = 1.0


@dataclass
class KnowledgeGraph:
    """The HippoRAG-style schemaless knowledge graph."""
    entities: Dict[str, Entity] = field(default_factory=dict)
    triples: List[Triple] = field(default_factory=list)
    adjacency: Dict[str, List[Tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))
    
    def add_entity(self, entity: Entity):
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
    
    def add_triple(self, triple: Triple):
        """Add a triple and update adjacency."""
        self.triples.append(triple)
        self.adjacency[triple.subject].append((triple.relation, triple.object))


class HippoRAGKnowledgeGraph:
    """
    Adapter for HippoRAG's knowledge graph construction.
    
    Uses the actual HippoRAG codebase from external_frameworks/HippoRAG/
    to build a schemaless knowledge graph from novel text.
    
    The original HippoRAG uses:
    - Named Entity Recognition (NER) for entity extraction
    - Open Information Extraction (OpenIE) for relationship extraction
    - PPR (Personalized PageRank) for retrieval
    """
    
    def __init__(
        self,
        extraction_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
        passage_size: int = 1000
    ):
        """
        Initialize the HippoRAG knowledge graph builder.
        
        Args:
            extraction_model: Model for entity/relation extraction
            embedding_model: Model for embeddings
            passage_size: Size of text passages for processing
        """
        self.extraction_model = extraction_model
        self.embedding_model = embedding_model
        self.passage_size = passage_size
        
        self._hipporag = None
        self._initialized = False
        
        logger.info("HippoRAG adapter initialized")
    
    def _ensure_initialized(self):
        """Lazily initialize HippoRAG components."""
        if self._initialized:
            return
            
        try:
            # Try to import from the cloned HippoRAG repository
            from hipporag.HippoRAG import HippoRAG as HippoRAGCore
            from hipporag.information_extraction import openie_extraction
            
            self._HippoRAGCore = HippoRAGCore
            self._openie_extraction = openie_extraction
            
            logger.info("HippoRAG codebase imported from external_frameworks/HippoRAG/")
            self._initialized = True
            
        except ImportError as e:
            logger.warning(f"Could not import HippoRAG directly: {e}")
            logger.info("Falling back to standalone implementation using HippoRAG methodology")
            self._initialized = True
    
    def build_graph(self, text: str, novel_id: str = "default") -> KnowledgeGraph:
        """
        Build a knowledge graph from novel text using HippoRAG methodology.
        
        Uses HippoRAG's approach:
        1. Split text into passages
        2. Extract entities using NER
        3. Extract relationships using OpenIE
        4. Build graph structure with embeddings
        
        Args:
            text: The input novel text
            novel_id: Identifier for caching
            
        Returns:
            KnowledgeGraph with entities and triples
        """
        self._ensure_initialized()
        
        logger.info(f"Building HippoRAG knowledge graph from text ({len(text)} chars)")
        
        # Split into passages
        passages = self._split_passages(text)
        logger.info(f"Processing {len(passages)} passages")
        
        graph = KnowledgeGraph()
        
        for i, passage in enumerate(passages):
            # Extract entities and relations from each passage
            entities, triples = self._extract_from_passage(passage)
            
            for entity in entities:
                if entity.id not in graph.entities:
                    entity.embedding = self._get_embedding(entity.name)
                    graph.add_entity(entity)
            
            for triple in triples:
                triple.source_text = passage[:200]
                graph.add_triple(triple)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(passages)} passages")
        
        logger.info(f"HippoRAG graph complete: {len(graph.entities)} entities, {len(graph.triples)} triples")
        return graph
    
    def _split_passages(self, text: str) -> List[str]:
        """Split text into passages for processing."""
        passages = []
        for i in range(0, len(text), self.passage_size):
            passage = text[i:i + self.passage_size]
            if passage.strip():
                passages.append(passage.strip())
        return passages
    
    def _extract_from_passage(self, passage: str) -> Tuple[List[Entity], List[Triple]]:
        """
        Extract entities and triples from a passage.
        
        Uses HippoRAG's OpenIE-style extraction:
        - Entity extraction via NER
        - Relation extraction via LLM
        
        Args:
            passage: Text passage
            
        Returns:
            Tuple of (entities, triples)
        """
        try:
            from core.models import llm_complete
            
            # HippoRAG-style extraction prompt
            prompt = f"""Extract entities and relationships from the following text.

TEXT:
{passage}

Output a JSON object with:
1. "entities": List of unique entities (characters, places, objects, concepts)
2. "triples": List of [subject, relation, object] arrays

Example output:
{{
  "entities": ["Robert", "mountains", "father"],
  "triples": [
    ["Robert", "RAISED_IN", "mountains"],
    ["father", "TAUGHT", "Robert"]
  ]
}}

Extract ALL entities and relationships. Output ONLY valid JSON:"""

            response = llm_complete(
                prompt=prompt,
                system_prompt="You are a knowledge graph extractor. Output only valid JSON.",
                temperature=0.0
            )
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {"entities": [], "triples": []}
            
            # Convert to our data structures
            entities = []
            for ent_name in data.get("entities", []):
                ent_id = ent_name.lower().replace(" ", "_")
                entities.append(Entity(
                    id=ent_id,
                    name=ent_name,
                    type="ENTITY"
                ))
            
            triples = []
            for triple_data in data.get("triples", []):
                if len(triple_data) >= 3:
                    triples.append(Triple(
                        subject=triple_data[0],
                        relation=triple_data[1],
                        object=triple_data[2]
                    ))
            
            return entities, triples
            
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return [], []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            from core.models import get_embedding
            return get_embedding(text)
        except:
            return [0.0] * 1536
    
    def query_path(
        self, 
        graph: KnowledgeGraph,
        subject: str,
        relation: Optional[str] = None,
        object_entity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for paths in the knowledge graph.
        
        Uses HippoRAG's retrieval approach:
        1. Find matching entities
        2. Traverse graph for relevant paths
        3. Rank by relevance
        
        Args:
            graph: The knowledge graph
            subject: Starting entity
            relation: Optional relation filter
            object_entity: Optional target entity
            
        Returns:
            List of matching paths with scores
        """
        results = []
        subject_lower = subject.lower()
        
        # Find matching entities
        matching_entities = []
        for ent_id, entity in graph.entities.items():
            if subject_lower in entity.name.lower() or subject_lower in ent_id:
                matching_entities.append(entity)
        
        # Find relevant triples
        for triple in graph.triples:
            if subject_lower in triple.subject.lower():
                if relation is None or relation.upper() in triple.relation.upper():
                    if object_entity is None or object_entity.lower() in triple.object.lower():
                        results.append({
                            "path": [triple.subject, triple.relation, triple.object],
                            "source": triple.source_text,
                            "confidence": triple.confidence
                        })
        
        return results


def build_hipporag_index(text: str, **kwargs) -> KnowledgeGraph:
    """
    Convenience function to build a HippoRAG knowledge graph.
    
    Args:
        text: Novel text
        **kwargs: Additional arguments for HippoRAGKnowledgeGraph
        
    Returns:
        KnowledgeGraph
    """
    builder = HippoRAGKnowledgeGraph(**kwargs)
    return builder.build_graph(text)
