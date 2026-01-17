"""
Synthetic Dataset Generator using Ollama
========================================

Generates synthetic backstory samples (consistent/contradict) 
using local Ollama models (llama3, mistral, etc.)

Usage:
    python data_generator.py --model llama3 --book "path/to/book.txt" --output data/synthetic.csv
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random
import re


@dataclass
class BackstorySample:
    """A single backstory sample."""
    id: int
    book_name: str
    char: str
    caption: str
    content: str
    label: str  # 'consistent' or 'contradict'


class OllamaClient:
    """Simple Ollama API client."""
    
    def __init__(self, model: str = "llama3"):
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama not running. Start with: ollama serve")
        except FileNotFoundError:
            raise RuntimeError("Ollama not installed. Install from: https://ollama.ai")
    
    def generate(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate text using Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "[Generation timed out]"
        except Exception as e:
            return f"[Error: {e}]"


class SyntheticDataGenerator:
    """
    Generates synthetic backstory samples using Ollama.
    
    Workflow:
    1. Extract character info from novel
    2. Generate consistent backstories
    3. Generate contradictory backstories (hard negatives)
    """
    
    def __init__(self, model: str = "llama3"):
        self.client = OllamaClient(model)
        self.samples: List[BackstorySample] = []
    
    def extract_characters(self, novel_text: str) -> List[str]:
        """Extract main character names from novel."""
        prompt = f"""Read this novel excerpt and list the 5 main character names only.
        
Novel excerpt (first 2000 chars):
{novel_text[:2000]}

Output format: Just the names, one per line. No explanations."""
        
        response = self.client.generate(prompt)
        # Parse names from response
        names = [line.strip() for line in response.split('\n') if line.strip() and len(line.strip()) < 50]
        return names[:5]  # Max 5 characters
    
    def generate_consistent_backstory(self, novel_text: str, character: str) -> str:
        """Generate a backstory that IS consistent with the novel."""
        prompt = f"""You are creating backstory for the character "{character}" from this novel.

Novel excerpt:
{novel_text[:3000]}

Write a 2-3 sentence backstory about {character}'s early life, childhood, or formative experiences that would be CONSISTENT with how they behave in the novel. 

Rules:
- Must not contradict any facts in the novel
- Should explain their personality/skills shown in the novel
- Be specific about events, places, dates

Output only the backstory, nothing else."""
        
        return self.client.generate(prompt)
    
    def generate_contradictory_backstory(self, novel_text: str, character: str) -> str:
        """Generate a backstory that CONTRADICTS the novel."""
        prompt = f"""You are creating a CONTRADICTORY backstory for "{character}" from this novel.

Novel excerpt:
{novel_text[:3000]}

Write a 2-3 sentence backstory about {character} that CONTRADICTS something about them in the novel.

Types of contradictions to use:
- Wrong family relationships (e.g., "had no siblings" when they do)
- Wrong skills/abilities (e.g., "never learned to swim" when they're a sailor)
- Wrong timeline (e.g., events happening before birth)
- Wrong nationality/origin
- Wrong personality traits

Output only the contradictory backstory, nothing else."""
        
        return self.client.generate(prompt)
    
    def generate_dataset(
        self, 
        novel_path: str, 
        book_name: str,
        samples_per_character: int = 5
    ) -> pd.DataFrame:
        """Generate a complete dataset for one novel."""
        
        # Load novel
        novel_path = Path(novel_path)
        if not novel_path.exists():
            raise FileNotFoundError(f"Novel not found: {novel_path}")
        
        novel_text = novel_path.read_text(encoding='utf-8', errors='replace')
        print(f"Loaded novel: {len(novel_text)} characters")
        
        # Extract characters
        print("Extracting characters...")
        characters = self.extract_characters(novel_text)
        print(f"Found characters: {characters}")
        
        samples = []
        sample_id = 1
        
        for char in characters:
            print(f"\nGenerating samples for: {char}")
            
            for i in range(samples_per_character):
                # Alternate between consistent and contradictory
                if i % 2 == 0:
                    content = self.generate_consistent_backstory(novel_text, char)
                    label = "consistent"
                else:
                    content = self.generate_contradictory_backstory(novel_text, char)
                    label = "contradict"
                
                samples.append({
                    'id': sample_id,
                    'book_name': book_name,
                    'char': char,
                    'caption': '',
                    'content': content,
                    'label': label
                })
                sample_id += 1
                print(f"  [{label}] {content[:60]}...")
        
        return pd.DataFrame(samples)
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save dataset in competition format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\nSaved {len(df)} samples to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic backstory dataset")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--book", required=True, help="Path to novel .txt file")
    parser.add_argument("--book-name", required=True, help="Book name for CSV")
    parser.add_argument("--output", default="data/synthetic_dataset.csv", help="Output CSV path")
    parser.add_argument("--samples-per-char", type=int, default=5, help="Samples per character")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(model=args.model)
    df = generator.generate_dataset(
        novel_path=args.book,
        book_name=args.book_name,
        samples_per_character=args.samples_per_char
    )
    generator.save_dataset(df, args.output)


if __name__ == "__main__":
    main()
