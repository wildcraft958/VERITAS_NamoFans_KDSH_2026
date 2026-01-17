
import os
import sys
import pandas as pd
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import json

# Add project root to path
sys.path.append(os.getcwd())

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

from src.pipeline import create_sota_pipeline

# Configuration
DATASET_DIR = "dataset"
BOOKS_DIR = os.path.join(DATASET_DIR, "Books")
TRAIN_FILE = os.path.join(DATASET_DIR, "train.csv")
OUTPUT_FILE = "calibration_results.json"

# Book filename mapping (CSV name -> Real filename)
BOOK_MAPPING = {
    "In Search of the Castaways": "In search of the castaways.txt",
    "The Count of Monte Cristo": "The Count of Monte Cristo.txt"
}

def load_novel(book_name):
    """Load novel text from file."""
    filename = BOOK_MAPPING.get(book_name)
    if not filename:
        print(f"Warning: No file found for {book_name}")
        return None
    
    path = os.path.join(BOOKS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: File not found at {path}")
        return None
        
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def run_calibration():
    print("🚀 Starting Threshold Calibration...")
    
    # Load dataset
    if not os.path.exists(TRAIN_FILE):
        print(f"Error: Train file not found at {TRAIN_FILE}")
        return

    df = pd.read_csv(TRAIN_FILE)
    print(f"Loaded {len(df)} samples from {TRAIN_FILE}")
    
    # Initialize results container
    results = []
    
    # Group by book to avoid re-ingesting (expensive!)
    grouped = df.groupby('book_name')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    

    # Group by book
    grouped = df.groupby('book_name')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        for book_name, group in tqdm(grouped, desc="Books"):
            tqdm.write(f"\n📖 Processing book: {book_name} ({len(group)} samples)")
            
            # Load novel text
            novel_text = load_novel(book_name)
            if not novel_text:
                continue
                
            # Initialize Pipeline
            pipeline = create_sota_pipeline(device=device)
            
            tqdm.write("  ⏳ Ingesting novel...")
            pipeline.ingest_novel(novel_text)
            
            # Process samples
            for _, row in tqdm(group.iterrows(), total=len(group), desc="Samples", leave=False):
                backstory = row['content']
                true_label = 1 if row['label'].lower() == 'consistent' else 0
                
                # Run check
                verdict = pipeline.check_consistency(backstory)
                
                scores = {
                    'nli': verdict.nli_score,
                    'perplexity': verdict.perplexity_score,
                    'temporal': verdict.temporal_score,
                    'evidence': 0.0
                }

                results.append({
                    "id": row['id'],
                    "book": book_name,
                    "label": true_label,
                    "total_score": verdict.confidence,
                    "is_consistent": verdict.consistent,
                    "scores": scores
                })
                
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted! Saving partial results...")
    
    finally:
        # Save raw results
        with open("calibration_raw_scores.json", "w") as f:
            json.dump(results, f, indent=2)
            
        analyze_results(results)


def analyze_results(results):
    print("\n📊 ANALYSIS & TUNING")
    
    if not results:
        print("No results to analyze.")
        return

    y_true = np.array([r['label'] for r in results])
    
    # 1. Total Score Tuning
    y_scores = np.array([r['total_score'] for r in results])
    
    best_f1 = 0
    best_thresh = 0.5
    
    thresholds = np.arange(0.1, 1.0, 0.01)
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"\n🏆 Optimal Consistency Threshold: {best_thresh:.2f}")
    print(f"   Max F1 Score: {best_f1:.4f}")
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        print(f"   AUC-ROC: {auc:.4f}")
    except:
        print("   AUC-ROC: N/A")
    
    print("\n✅ Calibration Complete. Update pipeline with these values.")

if __name__ == "__main__":
    run_calibration()
