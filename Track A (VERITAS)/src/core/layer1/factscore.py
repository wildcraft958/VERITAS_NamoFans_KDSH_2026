"""
FActScore-based atomic fact extraction.

Uses GPT-4 to decompose backstory text into independent, testable atomic facts.
Based on: https://github.com/shmsw25/FActScore (EMNLP 2023)
"""

import json
import re
from dataclasses import dataclass

from core.layer1.constraint_graph import AtomicFact
from core.models import llm_complete


FACT_EXTRACTION_PROMPT = """You are an expert at extracting atomic facts from text.

An atomic fact is a single, independent, testable piece of information that:
- Contains exactly ONE claim
- Can be verified independently of other facts
- Is as specific as possible
- Uses clear, unambiguous language

Given a character backstory, extract ALL atomic facts. Each fact should be a complete, standalone statement.

BACKSTORY:
{backstory}

Extract atomic facts as a JSON array. Each element should be an object with:
- "fact": The atomic fact as a complete sentence
- "source": The original sentence this fact came from

Example output format:
[
  {{"fact": "Robert was a young boy", "source": "Robert was a young boy raised in the mountains."}},
  {{"fact": "Robert was raised in the mountains", "source": "Robert was a young boy raised in the mountains."}}
]

IMPORTANT:
- Break down compound sentences into separate facts
- Include implicit facts that can be inferred
- Be exhaustive - extract every piece of information
- Each fact should be verifiable against the novel

Output ONLY the JSON array, no other text:"""


class FactExtractor:
    """
    Extracts atomic facts from backstory text using FActScore methodology.
    
    Based on the paper "FActScore: Fine-grained Atomic Evaluation of 
    Factual Precision in Long Form Text Generation" (EMNLP 2023).
    
    Repository: https://github.com/shmsw25/FActScore
    """
    
    def __init__(self, model: str | None = None):
        """
        Initialize the fact extractor.
        
        Args:
            model: Optional model override for extraction
        """
        self.model = model
    
    def extract(self, backstory: str) -> list[AtomicFact]:
        """
        Extract atomic facts from a backstory.
        
        Args:
            backstory: Character backstory text
            
        Returns:
            List of AtomicFact objects
        """
        prompt = FACT_EXTRACTION_PROMPT.format(backstory=backstory)
        
        response = llm_complete(
            prompt=prompt,
            system_prompt="You are a precise fact extraction system. Output only valid JSON.",
            model=self.model,
            temperature=0.0
        )
        
        # Parse JSON response
        facts_data = self._parse_response(response)
        
        # Convert to AtomicFact objects
        atoms = []
        for i, item in enumerate(facts_data):
            atom = AtomicFact(
                id=f"F{i+1}",
                text=item.get("fact", ""),
                source_sentence=item.get("source", "")
            )
            atoms.append(atom)
        
        return atoms
    
    def _parse_response(self, response: str) -> list[dict]:
        """Parse LLM response to extract JSON array."""
        # Try to find JSON array in response
        response = response.strip()
        
        # Handle markdown code blocks
        if "```json" in response:
            match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r"```\s*([\s\S]*?)\s*```", response)
            if match:
                response = match.group(1)
        
        # Try to parse as JSON
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data
            return []
        except json.JSONDecodeError:
            # Try to find array in response
            match = re.search(r"\[[\s\S]*\]", response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []
    
    def extract_with_sentences(self, backstory: str) -> tuple[list[AtomicFact], list[str]]:
        """
        Extract atomic facts and return source sentences separately.
        
        Args:
            backstory: Character backstory text
            
        Returns:
            Tuple of (atoms, source_sentences)
        """
        atoms = self.extract(backstory)
        sentences = list(set(a.source_sentence for a in atoms if a.source_sentence))
        return atoms, sentences
