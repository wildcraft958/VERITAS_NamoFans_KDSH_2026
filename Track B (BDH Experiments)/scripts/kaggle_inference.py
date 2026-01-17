"""
BDH SOTA Narrative Consistency - Kaggle H100 Inference
======================================================

Complete Kaggle notebook for KDSH 2026 Track B competition.

SOTA Components:
1. BDHScanner (Hebbian Learning + Synaptic Drift)
2. DocumentNLIClassifier (DeBERTa-v3)
3. NeuralEventExtractor (SpaCy + BERT-NER)
4. TemporalNarrativeReasoner (Allen's Algebra)
5. CrossEncoderEvidenceAttributor

Usage on Kaggle:
1. Upload BDH code as a Kaggle Dataset (name: "bdh-code")
2. Add competition dataset
3. Run this notebook

Dataset expected structure:
    /kaggle/input/kdsh-2026/
        ├── train.csv
        ├── test.csv
        └── Books/
            ├── Book1.txt
            └── Book2.txt
"""

# ============================================================================
# CELL 1: Setup & Dependencies
# ============================================================================

# Parse arguments (for flag-based invocation of optional features)
import argparse
import subprocess
import sys

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="BDH SOTA Narrative Consistency - Kaggle Inference"
    )
    parser.add_argument(
        '--use-pathway',
        action='store_true',
        help='Use Pathway for document ingestion and vector retrieval (optional)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )
    # Handle Kaggle/Jupyter notebook environment (no args)
    try:
        if 'ipykernel' in sys.modules:
            return argparse.Namespace(use_pathway=False, device='cuda')
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(use_pathway=False, device='cuda')
    return args

ARGS = parse_args()

def install_deps():
    """Install required packages."""
    packages = [
        "torch",
        "transformers",
        "sentence-transformers", 
        "spacy",
        "pandas",
        "tqdm",
        "seaborn",
        "matplotlib",
    ]
    
    # Optional: Pathway for streaming ingestion (only if flag enabled)
    if ARGS.use_pathway:
        packages.append("pathway-python")
        print("[Pathway] Adding pathway-python to dependencies")
    
    for pkg in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    
    # Download spaCy model
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Uncomment to run:
# install_deps()


import os
import sys
import torch
import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ============================================================================
# CELL 2: Add BDH Code to Path
# ============================================================================

# Kaggle paths
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# BDH code location (uploaded as Kaggle dataset)
BDH_CODE_PATH = KAGGLE_INPUT / "bdh-code" / "BDH"

# Alternative: If running locally or code is in working directory
if not BDH_CODE_PATH.exists():
    BDH_CODE_PATH = Path(".")  # Current directory
    
if not BDH_CODE_PATH.exists():
    BDH_CODE_PATH = Path("/kaggle/working/BDH")

# Add to Python path
sys.path.insert(0, str(BDH_CODE_PATH))
print(f"BDH Code Path: {BDH_CODE_PATH}")
print(f"Path exists: {BDH_CODE_PATH.exists()}")

# ============================================================================
# CELL 3: Device Setup
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    
    # H100 optimizations
    if "H100" in gpu_name:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("H100 optimizations enabled (TF32, cuDNN benchmark)")

# ============================================================================
# CELL 4: Load Dataset
# ============================================================================

# Competition dataset path
DATASET_PATH = KAGGLE_INPUT / "kdsh-2026-narrative-consistency"

# Alternative paths to try
DATASET_PATHS = [
    KAGGLE_INPUT / "kdsh-2026-narrative-consistency",
    KAGGLE_INPUT / "kdsh-2026",
    KAGGLE_INPUT / "kdsh2026",
    Path("dataset"),  # Local testing
]

# Find dataset
data_path = None
for p in DATASET_PATHS:
    if p.exists():
        data_path = p
        break

if data_path is None:
    print("WARNING: Competition dataset not found!")
    print("Checked paths:", [str(p) for p in DATASET_PATHS])
    data_path = Path("Dataset_kdsh")  # Fallback for local testing

print(f"Dataset path: {data_path}")

# Load CSVs
train_csv = data_path / "train.csv"
test_csv = data_path / "test.csv"
books_dir = data_path / "Books"

train_df = None
test_df = None

if train_csv.exists():
    train_df = pd.read_csv(train_csv)
    print(f"Train samples: {len(train_df)}")
    print(f"Columns: {train_df.columns.tolist()}")

if test_csv.exists():
    test_df = pd.read_csv(test_csv)
    print(f"Test samples: {len(test_df)}")

if books_dir.exists():
    books = list(books_dir.glob("*.txt"))
    print(f"Books found: {len(books)}")
    for b in books[:3]:
        print(f"  - {b.name}")

# ============================================================================
# CELL 5: Initialize SOTA Pipeline
# ============================================================================

print("\n" + "="*60)
print("INITIALIZING SOTA PIPELINE")
print("="*60)

from src.pipeline import create_sota_pipeline

# Create pipeline (loads all models)
pipeline = create_sota_pipeline(device=str(device))

print("\nPipeline ready!")

# ============================================================================
# CELL 5B: Optional Pathway Integration
# ============================================================================

# Initialize Pathway retriever if flag is enabled
pathway_retriever = None

if ARGS.use_pathway:
    print("\n[Pathway] Initializing Pathway retriever...")
    try:
        from src.pathway_integration import PATHWAY_AVAILABLE
        
        if PATHWAY_AVAILABLE:
            from src.pathway_integration import PathwayRetriever
            pathway_retriever = PathwayRetriever(
                use_pathway=True,
                chunk_size=300,
                chunk_overlap=50,
            )
            print("[Pathway] Retriever initialized (streaming mode)")
        else:
            # Fallback to simplified vector store
            from src.pathway_integration.retriever import PathwayRetriever
            pathway_retriever = PathwayRetriever(
                use_pathway=False,  # Uses numpy backend
                chunk_size=300,
                chunk_overlap=50,
            )
            print("[Pathway] Using simplified retriever (numpy backend)")
    except ImportError as e:
        print(f"[Pathway] Warning: Could not initialize retriever: {e}")
        print("[Pathway] Continuing with standard pipeline ingestion")


# ============================================================================
# CELL 6: Helper Functions
# ============================================================================

def load_book(book_name: str, books_dir: Path) -> str:
    """Load book text by name."""
    # Try exact match
    book_path = books_dir / f"{book_name}.txt"
    if book_path.exists():
        return book_path.read_text(encoding='utf-8', errors='replace')
    
    # Try case-insensitive search
    for f in books_dir.glob("*.txt"):
        if book_name.lower() in f.name.lower():
            return f.read_text(encoding='utf-8', errors='replace')
    
    # Try partial match
    for f in books_dir.glob("*.txt"):
        if any(word.lower() in f.name.lower() for word in book_name.split()):
            return f.read_text(encoding='utf-8', errors='replace')
    
    raise FileNotFoundError(f"Book not found: {book_name}")


def process_sample(row: pd.Series, pipeline, books_dir: Path, books_cache: dict) -> dict:
    """Process a single sample."""
    sample_id = row.get('id', row.name)
    book_name = row['book_name']
    backstory = row['content']
    
    # Load book (with caching)
    if book_name not in books_cache:
        try:
            novel_text = load_book(book_name, books_dir)
            books_cache[book_name] = novel_text
            
            # Ingest novel
            pipeline.ingest_novel(novel_text)
        except Exception as e:
            print(f"Error loading book {book_name}: {e}")
            return {
                'id': sample_id,
                'Prediction': 0,  # Default to contradict on error
                'Rationale': f"Error loading book: {e}"
            }
    else:
        # Re-use cached novel but still need to re-ingest
        # (pipeline state resets between samples)
        pipeline.ingest_novel(books_cache[book_name])
    
    # Check consistency
    try:
        verdict = pipeline.check_consistency(backstory)
        
        return {
            'id': sample_id,
            'Prediction': 1 if verdict.consistent else 0,
            'Confidence': verdict.confidence,
            'Rationale': verdict.rationale[:500] if hasattr(verdict, 'rationale') else ""
        }
    except Exception as e:
        print(f"Error checking consistency for {sample_id}: {e}")
        return {
            'id': sample_id,
            'Prediction': 0,
            'Rationale': f"Error: {e}"
        }

# ============================================================================
# CELL 7: Run Inference on Test Set
# ============================================================================

print("\n" + "="*60)
print("RUNNING INFERENCE")
print("="*60)

results = []
books_cache = {}

# Use test_df if available, otherwise train_df
df = test_df if test_df is not None else train_df

if df is not None:
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        result = process_sample(row, pipeline, books_dir, books_cache)
        results.append(result)
        
        # Progress logging
        if (idx + 1) % 10 == 0:
            preds = [r['Prediction'] for r in results]
            consistent = sum(preds)
            print(f"Progress: {idx+1}/{len(df)} | Consistent: {consistent} | Contradict: {len(preds)-consistent}")
else:
    print("No dataset loaded. Creating dummy result...")
    results = [{'id': 0, 'Prediction': 1, 'Rationale': 'No data'}]

# ============================================================================
# CELL 8: Save Results
# ============================================================================

print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Create results DataFrame
results_df = pd.DataFrame(results)

# Save CSV submission
submission_path = KAGGLE_WORKING / "submission.csv"
results_df[['id', 'Prediction']].to_csv(submission_path, index=False)
print(f"Submission saved to: {submission_path}")

# Save detailed results (for debugging)
detailed_path = KAGGLE_WORKING / "detailed_results.csv"
results_df.to_csv(detailed_path, index=False)
print(f"Detailed results saved to: {detailed_path}")

# Summary
print("\n" + "-"*40)
print("SUMMARY")
print("-"*40)
print(f"Total samples: {len(results)}")
print(f"Consistent: {sum(r['Prediction'] for r in results)}")
print(f"Contradict: {len(results) - sum(r['Prediction'] for r in results)}")

# ============================================================================
# CELL 9: Display Sample Results
# ============================================================================

print("\n" + "="*60)
print("SAMPLE RESULTS")
print("="*60)

# Show first 5 results
for r in results[:5]:
    label = "CONSISTENT" if r['Prediction'] == 1 else "CONTRADICT"
    print(f"\nID: {r['id']}")
    print(f"Prediction: {label}")
    if r.get('Rationale'):
        print(f"Rationale: {r['Rationale'][:200]}...")

# ============================================================================
# CELL 10: GPU Memory Report
# ============================================================================

if torch.cuda.is_available():
    print("\n" + "="*60)
    print("GPU MEMORY REPORT")
    print("="*60)
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
    print(f"Total: {total:.2f} GB")
    print(f"Free: {total - allocated:.2f} GB")

print("\n" + "="*60)
print("INFERENCE COMPLETE")
print("="*60)
