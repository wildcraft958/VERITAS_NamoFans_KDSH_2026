
import os
import sys
import pandas as pd
import torch
import json
from tqdm import tqdm

# Add project root to path
sys.path.append(os.getcwd())

from src.pipeline import create_sota_pipeline

# Configuration
DATASET_DIR = "dataset"
BOOKS_DIR = os.path.join(DATASET_DIR, "Books")
TEST_FILE = os.path.join(DATASET_DIR, "test.csv")
OUTPUT_FILE = "submission.csv"

# Book filename mapping
BOOK_MAPPING = {
    "In Search of the Castaways": "In search of the castaways.txt",
    "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
}

def load_novel(book_name):
    filename = BOOK_MAPPING.get(book_name)
    if not filename: return None
    path = os.path.join(BOOKS_DIR, filename)
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_submission():
    print("🚀 SOTA Narrative Consistency Generator")
    print("=======================================")
    
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found at {TEST_FILE}")
        return

    df = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df)} test samples")
    
    results = []
    
    # Check if we should use calibration results
    if os.path.exists("calibration_results.json"):
        print("Note: Found calibration data. (Update code manualy if needed)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    grouped = df.groupby('book_name')
    
    for book_name, group in tqdm(grouped, desc="Books"):
        tqdm.write(f"\n📖 Processing book: {book_name}")
        
        novel_text = load_novel(book_name)
        if not novel_text:
            print(f"  Warning: Skipping {book_name} (source not found)")
            continue
            
        # Create pipeline
        # Note: If you have tuned thresholds/weights, update them here!
        # Example: 
        # pipeline = create_sota_pipeline(device=device, nli_weight=0.4, perplexity_weight=0.2)
        pipeline = create_sota_pipeline(device=device)
        
        tqdm.write("  ⏳ Ingesting...")
        pipeline.ingest_novel(novel_text)
        
        for _, row in tqdm(group.iterrows(), total=len(group), desc="Samples", leave=False):
            backstory = row['content']
            
            # Detailed check
            verdict = pipeline.check_consistency(backstory, return_detailed=False)
            
            # Map verdict to label
            label = "consistent" if verdict.consistent else "contradict"
            
            results.append({
                "id": row['id'],
                "label": label
            })
            
    # Create prediction dataframe
    pred_df = pd.DataFrame(results)
    
    # Merge with original ID order
    final_df = df[['id']].merge(pred_df, on='id', how='left')
    
    # Fill missing?
    final_df['label'] = final_df['label'].fillna('consistent')
    
    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Submission saved to {OUTPUT_FILE}")
    print(final_df.head())

if __name__ == "__main__":
    generate_submission()
