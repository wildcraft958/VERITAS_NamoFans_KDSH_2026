"""
End-to-End Verification Script
=============================

Tests the full SOTA pipeline:
1. Pathway/File Ingestion (Simulated)
2. Global Coreference Resolution
3. Synaptic Drift Scan (with Viz)
4. NLI + Temporal checks
"""

import torch
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.sota_pipeline import create_sota_pipeline

def main():
    print("Initializing SOTA Pipeline...")
    pipeline = create_sota_pipeline(device="cpu") # Force CPU for test
    
    # 1. Ingest Novel (Simulated 2 books)
    novel_text = """
    Thalcave rode his horse across the pampas. He was a brave guide.
    Robert followed him. The river rose quickly.
    Later, Glenarvan arrived at the camp.
    Thalcave saved Robert from the condor in the Andes.
    """
    
    print("\n[Step 1] Ingesting Novel...")
    pipeline.ingest_novel(novel_text)
    
    # 2. Check Consistent Backstory
    print("\n[Step 2] Checking Consistent Backstory...")
    backstory_good = "Thalcave is a guide who saved Robert."
    verdict1 = pipeline.check_consistency(backstory_good)
    print(f"Verdict: {'✅' if verdict1.consistent else '❌'}")
    
    # 3. Check Inconsistent Backstory
    print("\n[Step 3] Checking Contradictory Backstory...")
    backstory_bad = "Thalcave abandoned Robert to the condor and never visited the Andes."
    verdict2 = pipeline.check_consistency(backstory_bad)
    print(f"Verdict: {'✅' if verdict2.consistent else '❌'}")
    
    print("\n[Step 4] Checking Viz Output...")
    import os
    if os.path.exists("hebbian_drift.png"):
        print("✅ Synaptic Drift visualization found (hebbian_drift.png).")
    else:
        print("❌ Visualization missing.")

if __name__ == "__main__":
    main()
