"""
Verify Real Dataset (KDSH 2026)
==============================

Runs the Hebbian BDH Pipeline on the provided dataset.
1. Reads `Dataset_kdsh/train.csv`.
2. Maps book names to `Dataset_kdsh/Books/*.txt`.
3. Ingests each book ONCE.
4. Checks all associated samples.
5. Reports accuracy and generates Synaptic Drift maps.
"""

import pandas as pd
import os
import glob
import sys
import os

# Add parent directory to path so we can import narrative_reasoning
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.sota_pipeline import create_sota_pipeline

# Path configuration
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, "dataset")
books_dir = os.path.join(dataset_dir, "Books")
train_csv = os.path.join(dataset_dir, "train.csv")

def find_book_file(book_name):
    """Finds the .txt file for a book name (case-insensitive)."""
    # Specific mapping or search
    if "Castaways" in book_name:
        return os.path.join(books_dir, "In search of the castaways.txt")
    if "Count of Monte Cristo" in book_name:
        return os.path.join(BOOKS_DIR, "The Count of Monte Cristo.txt")
    
    # Fallback: glob
    files = glob.glob(os.path.join(BOOKS_DIR, "*.txt"))
    for f in files:
        if book_name.lower() in f.lower():
            return f
    return None

def main():
    print("Initializing SOTA Pipeline...")
    pipeline = create_sota_pipeline(device="cuda") 
    
    print(f"Loading Dataset: {TRAIN_CSV}")
    df = pd.read_csv(TRAIN_CSV)
    
    # Group by book to minimize ingestion overhead
    grouped = df.groupby("book_name")
    
    total_samples = 0
    correct_predictions = 0
    
    for book_name, group in grouped:
        print(f"\n{'='*60}")
        print(f"Processing Book: {book_name}")
        print(f"Samples: {len(group)}")
        print(f"{'='*60}")
        
        # 1. Locate Book File
        book_path = find_book_file(book_name)
        if not book_path or not os.path.exists(book_path):
            print(f"❌ Book file not found for: {book_name}")
            continue
            
        # 2. Ingest Book (Hebbian Learning)
        print(f"Ingesting: {os.path.basename(book_path)}...")
        try:
            with open(book_path, 'r', encoding='utf-8') as f:
                novel_text = f.read()
        except UnicodeDecodeError:
             with open(book_path, 'r', encoding='latin-1') as f:
                novel_text = f.read()
                
        # Limit text for rapid testing (First 50k chars)
        print("  Truncating novel to 50k chars for fast verification...")
        novel_text = novel_text[:50000]
        
        pipeline.ingest_novel(novel_text)
        
        # 3. Process Samples
        for idx, row in group.iterrows():
            backstory = row['content']
            label = row['label'] # 'consistent' or 'contradict'
            
            print(f"\n[Sample #{row['id']}] Label: {label}")
            print(f"Backstory: {backstory[:100]}...")
            
            # Run check
            verdict = pipeline.check_consistency(backstory)
            
            # Map prediction
            pred_label = "consistent" if verdict.consistent else "contradict"
            is_correct = (pred_label == label)
            
            print(f"Verdict: {pred_label.upper()} (Drift: {verdict.scores.get('step_5_hebbian', {}).get('raw_drift', 0.0):.4f})")
            print(f"Result: {'✅ PASS' if is_correct else '❌ FAIL'}")
            
            total_samples += 1
            if is_correct:
                correct_predictions += 1
                
    # Final Report
    if total_samples > 0:
        acc = (correct_predictions / total_samples) * 100
        print(f"\n{'='*60}")
        print(f"Final Accuracy: {acc:.2f}% ({correct_predictions}/{total_samples})")
        print(f"{'='*60}")
    else:
        print("No samples processed.")

if __name__ == "__main__":
    main()
