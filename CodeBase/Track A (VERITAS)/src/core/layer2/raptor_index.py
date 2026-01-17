"""
RAPTOR hierarchical index for global narrative arc retrieval.

RAPTOR = Recursive Abstractive Processing for Tree-Organized Retrieval
Based on: https://github.com/profintegra/raptor-rag (ICLR 2024)
Paper: arXiv:2401.18059
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kdsh.config import settings
from kdsh.models import llm_complete, get_embedding


@dataclass
class RAPTORNode:
    """A node in the RAPTOR tree."""
    
    id: str
    level: int  # 0 = raw text, higher = more abstract
    text: str
    embedding: list[float] = field(default_factory=list)
    children: list[str] = field(default_factory=list)  # Child node IDs
    metadata: dict = field(default_factory=dict)


@dataclass
class RAPTORTree:
    """The full RAPTOR tree structure."""
    
    nodes: dict[str, RAPTORNode] = field(default_factory=dict)
    root_ids: list[str] = field(default_factory=list)
    max_level: int = 0


SUMMARIZE_PROMPT = """Summarize the following text chunks into a coherent summary.
Preserve key narrative elements: characters, events, relationships, timeline.

TEXT CHUNKS:
{chunks}

Provide a concise summary (2-3 sentences) that captures the essential information:"""


class RAPTORIndex:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
    
    Builds a hierarchical tree of recursive summaries from a novel,
    enabling retrieval at multiple abstraction levels.
    
    Repository: https://github.com/profintegra/raptor-rag
    
    Levels:
    - Level 0: Raw text chunks (512 tokens)
    - Level 1: Chunk summaries (groups of ~5 chunks)
    - Level 2: Episode summaries (groups of summaries)
    - Level 3: Arc summaries (global narrative)
    """
    
    def __init__(
        self,
        novel_text: str | None = None,
        chunk_size: int | None = None,
        cache_dir: str = ".raptor_cache"
    ):
        """
        Initialize RAPTOR index.
        
        Args:
            novel_text: Full novel text (optional, can call build() later)
            chunk_size: Size of base chunks in characters
            cache_dir: Directory to cache index
        """
        self.chunk_size = chunk_size or settings.retriever.raptor_chunk_size
        self.cache_dir = Path(cache_dir)
        self.tree = RAPTORTree()
        
        if novel_text:
            self.build(novel_text)
    
    def build(self, novel_text: str, novel_id: str = "default") -> None:
        """
        Build the RAPTOR tree from novel text.
        
        Args:
            novel_text: Full novel text
            novel_id: Identifier for caching
        """
        # Check cache
        cache_path = self._get_cache_path(novel_id, novel_text)
        if cache_path.exists():
            self._load_cache(cache_path)
            return
        
        # Level 0: Chunk the text
        chunks = self._chunk_text(novel_text)
        level0_nodes = []
        
        for i, chunk in enumerate(chunks):
            node = RAPTORNode(
                id=f"L0_{i}",
                level=0,
                text=chunk,
                metadata={"chunk_index": i}
            )
            self.tree.nodes[node.id] = node
            level0_nodes.append(node)
        
        # Build higher levels recursively
        current_level_nodes = level0_nodes
        level = 0
        
        while len(current_level_nodes) > 1:
            level += 1
            next_level_nodes = self._build_level(current_level_nodes, level)
            current_level_nodes = next_level_nodes
        
        # Set root nodes
        self.tree.root_ids = [n.id for n in current_level_nodes]
        self.tree.max_level = level
        
        # Generate embeddings for all nodes
        self._embed_all_nodes()
        
        # Cache the tree
        self._save_cache(cache_path)
    
    def _chunk_text(self, text: str) -> list[str]:
        """Split text into chunks of approximately chunk_size characters."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _build_level(
        self, 
        nodes: list[RAPTORNode], 
        level: int,
        group_size: int = 5
    ) -> list[RAPTORNode]:
        """Build a higher level by summarizing groups of nodes."""
        new_nodes = []
        
        for i in range(0, len(nodes), group_size):
            group = nodes[i:i + group_size]
            
            # Combine texts
            combined_text = "\n\n---\n\n".join([n.text for n in group])
            
            # Summarize
            summary = self._summarize(combined_text)
            
            # Create new node
            node = RAPTORNode(
                id=f"L{level}_{i // group_size}",
                level=level,
                text=summary,
                children=[n.id for n in group],
                metadata={"source_nodes": len(group)}
            )
            
            self.tree.nodes[node.id] = node
            new_nodes.append(node)
        
        return new_nodes
    
    def _summarize(self, text: str) -> str:
        """Generate a summary using LLM."""
        prompt = SUMMARIZE_PROMPT.format(chunks=text[:8000])  # Truncate if too long
        
        return llm_complete(
            prompt=prompt,
            system_prompt="You are a skilled summarizer. Be concise but preserve key narrative details.",
            temperature=0.0,
            max_tokens=500
        )
    
    def _embed_all_nodes(self) -> None:
        """Generate embeddings for all nodes."""
        for node_id, node in self.tree.nodes.items():
            if not node.embedding:
                node.embedding = get_embedding(node.text[:8000])
    
    def query(
        self, 
        query: str, 
        top_k: int | None = None,
        min_level: int = 0,
        max_level: int | None = None
    ) -> list[tuple[RAPTORNode, float]]:
        """
        Query the RAPTOR index at multiple levels.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_level: Minimum abstraction level to search
            max_level: Maximum abstraction level to search
            
        Returns:
            List of (node, similarity_score) tuples
        """
        top_k = top_k or settings.retriever.raptor_top_k
        max_level = max_level if max_level is not None else self.tree.max_level
        
        # Get query embedding
        query_embedding = get_embedding(query)
        
        # Score all nodes in range
        scores = []
        for node_id, node in self.tree.nodes.items():
            if min_level <= node.level <= max_level:
                if node.embedding:
                    sim = self._cosine_similarity(query_embedding, node.embedding)
                    scores.append((node, sim))
        
        # Sort and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    def _get_cache_path(self, novel_id: str, novel_text: str) -> Path:
        """Get cache path for this novel."""
        text_hash = hashlib.md5(novel_text[:10000].encode()).hexdigest()[:8]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"raptor_{novel_id}_{text_hash}.json"
    
    def _save_cache(self, path: Path) -> None:
        """Save tree to cache."""
        data = {
            "max_level": self.tree.max_level,
            "root_ids": self.tree.root_ids,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "level": node.level,
                    "text": node.text,
                    "embedding": node.embedding,
                    "children": node.children,
                    "metadata": node.metadata
                }
                for node_id, node in self.tree.nodes.items()
            }
        }
        with open(path, "w") as f:
            json.dump(data, f)
    
    def _load_cache(self, path: Path) -> None:
        """Load tree from cache."""
        with open(path) as f:
            data = json.load(f)
        
        self.tree.max_level = data["max_level"]
        self.tree.root_ids = data["root_ids"]
        self.tree.nodes = {}
        
        for node_id, node_data in data["nodes"].items():
            self.tree.nodes[node_id] = RAPTORNode(
                id=node_data["id"],
                level=node_data["level"],
                text=node_data["text"],
                embedding=node_data["embedding"],
                children=node_data["children"],
                metadata=node_data["metadata"]
            )
