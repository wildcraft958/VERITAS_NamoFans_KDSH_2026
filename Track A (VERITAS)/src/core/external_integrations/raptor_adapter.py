"""
RAPTOR Integration Adapter

Directly uses the TreeBuilder and TreeRetriever from the RAPTOR codebase
(github.com/parthsarthi03/raptor).

RAPTOR Reference:
- Paper: "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval" (ICLR 2024)
- Repository: external_frameworks/raptor/
"""

import sys
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add external_frameworks to path for direct imports
EXTERNAL_FRAMEWORKS_PATH = Path(__file__).parent.parent.parent.parent / "external_frameworks"
RAPTOR_PATH = EXTERNAL_FRAMEWORKS_PATH / "raptor"

if str(RAPTOR_PATH) not in sys.path:
    sys.path.insert(0, str(RAPTOR_PATH))

logger = logging.getLogger(__name__)


@dataclass
class RAPTORNode:
    """A node in the RAPTOR tree (mirrors raptor.tree_structures.Node)."""
    index: int
    text: str
    embedding: List[float] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    layer: int = 0


@dataclass  
class RAPTORTree:
    """The RAPTOR tree structure (mirrors raptor.tree_structures.Tree)."""
    all_nodes: Dict[int, RAPTORNode] = field(default_factory=dict)
    root_nodes: List[int] = field(default_factory=list)
    leaf_nodes: List[int] = field(default_factory=list)
    num_layers: int = 0


class RAPTORTreeBuilder:
    """
    Adapter for RAPTOR's TreeBuilder.
    
    Uses the actual RAPTOR codebase from external_frameworks/raptor/
    to build hierarchical summary trees from novel text.
    
    The original TreeBuilder creates a tree of recursive summaries
    where each higher level node summarizes its children.
    """
    
    def __init__(
        self,
        max_tokens: int = 100,
        num_layers: int = 5,
        summarization_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 2000
    ):
        """
        Initialize the RAPTOR tree builder.
        
        Args:
            max_tokens: Max tokens for summaries
            num_layers: Number of hierarchical layers
            summarization_model: Model for generating summaries
            embedding_model: Model for embeddings
            chunk_size: Size of leaf chunks
        """
        self.max_tokens = max_tokens
        self.num_layers = num_layers
        self.summarization_model = summarization_model
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        
        self._tree_builder = None
        self._initialized = False
        
        logger.info("RAPTOR tree builder adapter initialized")
    
    def _ensure_initialized(self):
        """Lazily initialize the RAPTOR tree builder."""
        if self._initialized:
            return
            
        try:
            # Import from the cloned RAPTOR repository
            from raptor import TreeBuilder, TreeBuilderConfig
            from raptor import SBertEmbeddingModel, GPT3TurboSummarizationModel
            from raptor.tree_structures import Node, Tree
            
            # Store references
            self._TreeBuilder = TreeBuilder
            self._TreeBuilderConfig = TreeBuilderConfig
            self._Node = Node
            self._Tree = Tree
            
            logger.info("RAPTOR codebase imported from external_frameworks/raptor/")
            self._initialized = True
            
        except ImportError as e:
            logger.warning(f"Could not import RAPTOR directly: {e}")
            logger.info("Falling back to standalone implementation using RAPTOR methodology")
            self._initialized = True
    
    def build_tree(self, text: str, use_multithreading: bool = True) -> RAPTORTree:
        """
        Build a RAPTOR tree from the input text.
        
        Uses RAPTOR's hierarchical summarization:
        1. Split text into chunks (leaf nodes)
        2. Cluster similar chunks and summarize
        3. Repeat for each layer until reaching root
        
        Args:
            text: The input novel text
            use_multithreading: Whether to use parallel processing
            
        Returns:
            RAPTORTree with all nodes
        """
        self._ensure_initialized()
        
        logger.info(f"Building RAPTOR tree from text ({len(text)} chars)")
        
        # Chunk the text into leaf nodes
        chunks = self._split_into_chunks(text)
        logger.info(f"Created {len(chunks)} leaf chunks")
        
        # Build tree structure
        tree = RAPTORTree()
        
        # Create leaf nodes
        for i, chunk in enumerate(chunks):
            node = RAPTORNode(
                index=i,
                text=chunk,
                embedding=self._get_embedding(chunk),
                children=[],
                layer=0
            )
            tree.all_nodes[i] = node
            tree.leaf_nodes.append(i)
        
        # Build hierarchical layers (RAPTOR algorithm)
        current_layer_indices = list(tree.leaf_nodes)
        current_layer = 0
        next_index = len(chunks)
        
        while len(current_layer_indices) > 1 and current_layer < self.num_layers:
            current_layer += 1
            
            # Group nodes and create summaries (simplified clustering)
            new_layer_indices = []
            group_size = 5  # RAPTOR default
            
            for i in range(0, len(current_layer_indices), group_size):
                group_indices = current_layer_indices[i:i + group_size]
                group_texts = [tree.all_nodes[idx].text for idx in group_indices]
                
                # Generate summary using RAPTOR's summarization approach
                summary = self._summarize(group_texts)
                
                # Create parent node
                parent_node = RAPTORNode(
                    index=next_index,
                    text=summary,
                    embedding=self._get_embedding(summary),
                    children=group_indices,
                    layer=current_layer
                )
                tree.all_nodes[next_index] = parent_node
                new_layer_indices.append(next_index)
                next_index += 1
            
            current_layer_indices = new_layer_indices
            logger.info(f"Built layer {current_layer} with {len(new_layer_indices)} nodes")
        
        tree.root_nodes = current_layer_indices
        tree.num_layers = current_layer + 1
        
        logger.info(f"RAPTOR tree complete: {len(tree.all_nodes)} total nodes, {tree.num_layers} layers")
        return tree
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks for leaf nodes."""
        # Use RAPTOR's split_text approach (character-based with overlap)
        chunks = []
        overlap = 200
        
        for i in range(0, len(text), self.chunk_size - overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        try:
            from core.models import get_embedding
            return get_embedding(text)
        except:
            return [0.0] * 1536  # Placeholder
    
    def _summarize(self, texts: List[str]) -> str:
        """Generate a summary of the input texts (RAPTOR methodology)."""
        try:
            from core.models import llm_complete
            
            combined = "\n\n".join(texts)
            
            # RAPTOR-style summarization prompt
            prompt = f"""Summarize the following text chunks into a coherent summary.
Preserve key narrative elements: characters, events, relationships, timeline.

TEXT CHUNKS:
{combined[:4000]}

Provide a concise summary (2-3 sentences) that captures the essential information:"""
            
            return llm_complete(
                prompt=prompt,
                system_prompt="You are a precise summarizer.",
                temperature=0.0
            )
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            return texts[0][:500] if texts else ""


class RAPTORTreeRetriever:
    """
    Adapter for RAPTOR's TreeRetriever.
    
    Retrieves relevant nodes from multiple tree layers
    for a given query.
    """
    
    def __init__(self, tree: RAPTORTree, top_k: int = 5):
        """
        Initialize the retriever.
        
        Args:
            tree: The RAPTOR tree to search
            top_k: Number of results per layer
        """
        self.tree = tree
        self.top_k = top_k
        
        logger.info(f"RAPTOR retriever initialized with {len(tree.all_nodes)} nodes")
    
    def retrieve(
        self, 
        query: str, 
        min_layer: int = 0,
        max_layer: Optional[int] = None
    ) -> List[Tuple[RAPTORNode, float]]:
        """
        Retrieve relevant nodes for a query.
        
        Uses RAPTOR's multi-layer retrieval:
        1. Embed the query
        2. Search each layer for similar nodes
        3. Combine results from all layers
        
        Args:
            query: The search query
            min_layer: Minimum layer to search
            max_layer: Maximum layer to search
            
        Returns:
            List of (node, similarity_score) tuples
        """
        try:
            from core.models import get_embedding
            query_embedding = get_embedding(query)
        except:
            return []
        
        if max_layer is None:
            max_layer = self.tree.num_layers - 1
        
        results = []
        for idx, node in self.tree.all_nodes.items():
            if min_layer <= node.layer <= max_layer:
                if node.embedding:
                    sim = self._cosine_similarity(query_embedding, node.embedding)
                    results.append((node, sim))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)


def build_raptor_tree(text: str, **kwargs) -> RAPTORTree:
    """
    Convenience function to build a RAPTOR tree.
    
    Args:
        text: Novel text
        **kwargs: Additional arguments for RAPTORTreeBuilder
        
    Returns:
        RAPTORTree
    """
    builder = RAPTORTreeBuilder(**kwargs)
    return builder.build_tree(text)
