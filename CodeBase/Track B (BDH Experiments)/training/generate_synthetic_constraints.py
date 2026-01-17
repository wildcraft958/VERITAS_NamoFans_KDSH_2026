"""
Generate Synthetic Constraints (Ollama Powered)
==============================================

Generates hard negative samples (contradictions) using local LLMs via Ollama.
Designed to align with the KDSH 2026 dataset schema.

Process:
1. Reads `dataset/train.csv` (Consistent samples)
2. Uses Ollama to generate subtle contradictions
3. Saves in standard format for pipeline training/calibration

Usage:
    python training/generate_synthetic_constraints.py --model mistral

Note: Uses shared OllamaClient from scripts/data_generator.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import os
import argparse
from tqdm import tqdm

# Import shared Ollama client
try:
    from scripts.data_generator import OllamaClient
except ImportError:
    # Fallback: Define simple client if import fails
    import subprocess
    
    class OllamaClient:
        def __init__(self, model: str = "mistral"):
            self.model = model
        
        def generate(self, prompt: str) -> str:
            try:
                result = subprocess.run(
                    ["ollama", "run", self.model, prompt],
                    capture_output=True, text=True, timeout=120
                )
                return result.stdout.strip()
            except Exception as e:
                return f"[Error: {e}]"

# Configuration
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_FILE = DATASET_DIR / "train.csv"
OUTPUT_FILE = DATASET_DIR / "synthetic_constraints.csv"


def generate_contradiction(client: OllamaClient, claim: str) -> str:
    """Generate a contradictory version of a claim."""
    prompt = (
        f"Rewrite the following backstory claim to be subtly contradictory or factually incorrect "
        f"regarding the same subject. Do not change the topic, just flip a key fact (e.g., location, action, outcome). "
        f"Output ONLY the rewritten text, no explanations.\n\n"
        f"Original Claim: {claim}\n"
        f"Contradictory Claim:"
    )
    return client.generate(prompt)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic constraints using Ollama")
    parser.add_argument("--model", type=str, default="mistral", help="Ollama model to use (default: mistral)")
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to generate (default: 10 for demo)")
    args = parser.parse_args()
    
    print(f"🚀 Synthetic Data Generator (Model: {args.model})")
    
    if not TRAIN_FILE.exists():
        print(f"Error: {TRAIN_FILE} not found.")
        return

    # Initialize Ollama client
    try:
        client = OllamaClient(model=args.model)
    except Exception as e:
        print(f"Error initializing Ollama: {e}")
        return

    # Load consistent samples only
    df = pd.read_csv(TRAIN_FILE)
    df_consistent = df[df['label'].str.lower() == 'consistent']
    
    if df_consistent.empty:
        print("No consistent samples found in training data.")
        return

    print(f"Loaded {len(df_consistent)} consistent samples. Generating {args.limit} negatives...")
    
    new_samples = []
    
    # Iterate and generate
    count = 0
    for _, row in tqdm(df_consistent.iterrows(), total=min(len(df_consistent), args.limit)):
        if count >= args.limit:
            break
            
        original_claim = row['content']
        book_name = row['book_name']
        original_id = row['id']
        
        # Generate contradiction
        contradiction = generate_contradiction(client, original_claim)
        
        if contradiction and contradiction != original_claim and not contradiction.startswith("[Error"):
            # Create new row matching dataset schema
            new_samples.append({
                "id": f"{original_id}_neg",
                "book_name": book_name,
                "content": contradiction,
                "label": "contradict"
            })
            count += 1
            
    if not new_samples:
        print("Failed to generate samples. Check Ollama connection/model.")
        return

    # Create DataFrame
    synthetic_df = pd.DataFrame(new_samples)
    
    # combine with original if desired, or save separate
    synthetic_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n✅ Generated {len(synthetic_df)} synthetic constraints")
    print(f"   Saved to: {OUTPUT_FILE}")
    print("\nSample Preview:")
    print(synthetic_df.head())

if __name__ == "__main__":
    main()

