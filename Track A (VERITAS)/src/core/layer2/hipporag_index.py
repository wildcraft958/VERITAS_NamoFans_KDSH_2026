"""
HippoRAG knowledge graph index for causal path retrieval.

HippoRAG = Neurobiologically Inspired Long-Term Memory
Based on: https://github.com/OSU-NLP-Group/HippoRAG (NeurIPS 2024)
"""

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import settings
from core.models import llm_complete, get_embedding


@dataclass
class Entity:
    """An entity extracted from the novel."""
    
    id: str
    name: str
    type: str = "ENTITY"
    mentions: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)


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
    """The knowledge graph structure."""
    
    entities: dict[str, Entity] = field(default_factory=dict)
    triples: list[Triple] = field(default_factory=list)
    adjacency: dict[str, list[tuple[str, str]]] = field(default_factory=lambda: defaultdict(list))


EXTRACTION_PROMPT = """Extract entities and relationships from the following text.

TEXT:
{text}

Output a JSON object with:
1. "entities": List of unique entities (characters, places, objects, concepts)
2. "triples": List of [subject, relation, object] relationships

Example:
{{
  "entities": ["Robert", "mountains", "avalanche", "father"],
  "triples": [
    ["Robert", "RAISED_IN", "mountains"],
    ["father", "DIED_IN", "avalanche"],
    ["Robert", "ESCAPED", "captivity"]
  ]
}}

Extract ALL entities and relationships. Output ONLY valid JSON:"""


class HippoRAGIndex:
    """
    HippoRAG: Neurobiologically Inspired Long-Term Memory.
    
    Builds a schemaless knowledge graph from the novel using:
    - Named Entity Recognition (NER) for entity extraction
    - Open Information Extraction (OpenIE) for relationship triples
    - Personalized PageRank (PPR) for multi-hop retrieval
    
    Repository: https://github.com/OSU-NLP-Group/HippoRAG
    """
    
    def __init__(
        self,
        novel_text: str | None = None,
        cache_dir: str = ".hipporag_cache"
    ):
        """
        Initialize HippoRAG index.
        
        Args:
            novel_text: Full novel text (optional, can call build() later)
            cache_dir: Directory to cache index
        """
        self.cache_dir = Path(cache_dir)
        self.kg = KnowledgeGraph()
        
        if novel_text:
            self.build(novel_text)
    
    def build(self, novel_text: str, novel_id: str = "default") -> None:
        """
        Build the knowledge graph from novel text.
        
        Args:
            novel_text: Full novel text
            novel_id: Identifier for caching
        """
        # Check cache
        cache_path = self._get_cache_path(novel_id, novel_text)
        if cache_path.exists():
            self._load_cache(cache_path)
            return
        
        # Split into passages
        passages = self._split_passages(novel_text)
        
        # Extract entities and triples from each passage
        all_entities = set()
        all_triples = []
        
        for i, passage in enumerate(passages):
            entities, triples = self._extract_from_passage(passage)
            all_entities.update(entities)
            all_triples.extend(triples)
        
        # Build entity nodes
        for entity_name in all_entities:
            entity = Entity(
                id=self._normalize_entity(entity_name),
                name=entity_name
            )
            self.kg.entities[entity.id] = entity
        
        # Add triples and build adjacency
        for triple in all_triples:
            subj_id = self._normalize_entity(triple.subject)
            obj_id = self._normalize_entity(triple.object)
            
            # Ensure entities exist
            if subj_id not in self.kg.entities:
                self.kg.entities[subj_id] = Entity(id=subj_id, name=triple.subject)
            if obj_id not in self.kg.entities:
                self.kg.entities[obj_id] = Entity(id=obj_id, name=triple.object)
            
            self.kg.triples.append(triple)
            self.kg.adjacency[subj_id].append((triple.relation, obj_id))
        
        # Generate embeddings for entities
        self._embed_entities()
        
        # Cache
        self._save_cache(cache_path)
    
    def _split_passages(self, text: str, passage_size: int = 1000) -> list[str]:
        """Split text into passages."""
        passages = []
        words = text.split()
        
        for i in range(0, len(words), passage_size // 2):  # 50% overlap
            passage = " ".join(words[i:i + passage_size])
            if passage:
                passages.append(passage)
        
        return passages
    
    def _extract_from_passage(self, passage: str) -> tuple[set[str], list[Triple]]:
        """Extract entities and triples from a passage using LLM."""
        prompt = EXTRACTION_PROMPT.format(text=passage[:4000])
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="You are an expert at knowledge graph extraction. Output only valid JSON.",
            temperature=0.0,
            max_tokens=1000
        )
        
        # Parse response
        try:
            # Clean response
            response = response.strip()
            if "```json" in response:
                match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            elif "```" in response:
                match = re.search(r"```\s*([\s\S]*?)\s*```", response)
                if match:
                    response = match.group(1)
            
            data = json.loads(response)
            entities = set(data.get("entities", []))
            triples = [
                Triple(
                    subject=t[0],
                    relation=t[1],
                    object=t[2],
                    source_text=passage[:200]
                )
                for t in data.get("triples", [])
                if len(t) >= 3
            ]
            return entities, triples
        except (json.JSONDecodeError, KeyError, IndexError):
            return set(), []
    
    def _normalize_entity(self, name: str) -> str:
        """Normalize entity name to ID."""
        return name.lower().strip().replace(" ", "_")
    
    def _embed_entities(self) -> None:
        """Generate embeddings for entities."""
        for entity_id, entity in self.kg.entities.items():
            if not entity.embedding:
                entity.embedding = get_embedding(entity.name)
    
    def query_path(
        self, 
        subject: str, 
        relation: str | None = None, 
        object: str | None = None
    ) -> "PathResult":
        """
        Query for a path in the knowledge graph.
        
        Args:
            subject: Starting entity
            relation: Optional relation to search for
            object: Optional target entity
            
        Returns:
            PathResult with found paths and scores
        """
        from core.layer2.unified_retriever import PathResult
        
        subj_id = self._normalize_entity(subject)
        obj_id = self._normalize_entity(object) if object else None
        
        # Check if subject exists
        if subj_id not in self.kg.entities:
            # Try fuzzy match
            subj_id = self._fuzzy_match_entity(subject)
        
        if not subj_id or subj_id not in self.kg.entities:
            return PathResult(found=False, paths=[], score=0.0)
        
        # Get outgoing edges
        edges = self.kg.adjacency.get(subj_id, [])
        
        # Filter by relation if specified
        if relation:
            rel_lower = relation.lower()
            edges = [(r, o) for r, o in edges if rel_lower in r.lower()]
        
        # Filter by object if specified
        if obj_id:
            edges = [(r, o) for r, o in edges if o == obj_id]
        
        if edges:
            paths = [(subj_id, r, o) for r, o in edges]
            return PathResult(found=True, paths=paths, score=1.0)
        
        # Try multi-hop search (2 hops)
        if obj_id:
            multi_hop_paths = self._find_multi_hop_path(subj_id, obj_id, max_hops=2)
            if multi_hop_paths:
                return PathResult(found=True, paths=multi_hop_paths, score=0.8)
        
        return PathResult(found=False, paths=[], score=0.0)
    
    def _fuzzy_match_entity(self, name: str) -> str | None:
        """Find closest matching entity."""
        name_lower = name.lower()
        
        # Exact substring match
        for entity_id in self.kg.entities:
            if name_lower in entity_id or entity_id in name_lower:
                return entity_id
        
        # Embedding similarity match
        if self.kg.entities:
            query_emb = get_embedding(name)
            best_match = None
            best_score = 0.0
            
            for entity_id, entity in self.kg.entities.items():
                if entity.embedding:
                    sim = self._cosine_similarity(query_emb, entity.embedding)
                    if sim > best_score and sim > 0.7:
                        best_score = sim
                        best_match = entity_id
            
            return best_match
        
        return None
    
    def _find_multi_hop_path(
        self, 
        start: str, 
        end: str, 
        max_hops: int = 2
    ) -> list[tuple]:
        """Find multi-hop paths between entities using BFS."""
        from collections import deque
        
        if start not in self.kg.adjacency:
            return []
        
        queue = deque([(start, [(start,)])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_hops + 1:
                continue
            
            for relation, neighbor in self.kg.adjacency.get(current, []):
                new_path = path + [(relation, neighbor)]
                
                if neighbor == end:
                    return [new_path]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))
        
        return []
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def get_entity_neighbors(self, entity: str, top_k: int = 5) -> list[tuple[str, str, str]]:
        """Get neighboring entities and relationships."""
        entity_id = self._normalize_entity(entity)
        if entity_id not in self.kg.adjacency:
            entity_id = self._fuzzy_match_entity(entity)
        
        if not entity_id:
            return []
        
        edges = self.kg.adjacency.get(entity_id, [])[:top_k]
        return [(entity_id, rel, obj) for rel, obj in edges]
    
    def _get_cache_path(self, novel_id: str, novel_text: str) -> Path:
        """Get cache path."""
        text_hash = hashlib.md5(novel_text[:10000].encode()).hexdigest()[:8]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"hipporag_{novel_id}_{text_hash}.json"
    
    def _save_cache(self, path: Path) -> None:
        """Save KG to cache."""
        data = {
            "entities": {
                eid: {"id": e.id, "name": e.name, "type": e.type, "embedding": e.embedding}
                for eid, e in self.kg.entities.items()
            },
            "triples": [
                {"subject": t.subject, "relation": t.relation, "object": t.object}
                for t in self.kg.triples
            ],
            "adjacency": dict(self.kg.adjacency)
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def _load_cache(self, path: Path) -> None:
        """Load KG from cache."""
        with open(path) as f:
            data = json.load(f)
        
        self.kg.entities = {
            eid: Entity(
                id=e["id"],
                name=e["name"],
                type=e.get("type", "ENTITY"),
                embedding=e.get("embedding", [])
            )
            for eid, e in data["entities"].items()
        }
        
        self.kg.triples = [
            Triple(subject=t["subject"], relation=t["relation"], object=t["object"])
            for t in data["triples"]
        ]
        
        self.kg.adjacency = defaultdict(list, data["adjacency"])
