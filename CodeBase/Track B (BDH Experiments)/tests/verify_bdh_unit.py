"""
Verify BDH Scanner (Unit Test)
=============================

Tests the BDHScanner math (Hebbian + Sparse) in isolation.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.signals.bdh_scanner import BDHScanner
import numpy as np

def test_bdh_hebbian_logic():
    print("Testing BDHScanner Hebbian Logic...")
    
    dim = 64
    scanner = BDHScanner(hidden_dim=dim, decay=0.9, sparsity=0.1, device='cpu')
    
    # 1. Create Synthetic "Concept" vectors
    # Concept A: "Thalcave is Guide"
    concept_A = torch.randn(dim)
    concept_A = concept_A / torch.norm(concept_A)
    
    # Concept B: "Robert is Saved"
    concept_B = torch.randn(dim)
    concept_B = concept_B / torch.norm(concept_B)
    
    # 2. Train (Prime) on Concept A (Hebbian Learning)
    print("\n[Phase 1] Priming Synapses with Concept A...")
    # Repeat a few times to strengthen connection
    drift_trace = []
    for _ in range(5):
        d, _ = scanner.step(concept_A, concept_A)
        drift_trace.append(d)
    
    print(f"  Drift Trace (Training): {[f'{x:.4f}' for x in drift_trace]}")
    # Drift should decrease as it learns self-association
    if drift_trace[-1] < drift_trace[0]:
        print("  ✅ Hebbian Learning Confirmed: Drift decreased over time.")
    else:
        print("  ❌ Hebbian Learning Failed: Drift did not decrease.")

    # 3. Test Consistent Input (Concept A)
    print("\n[Phase 2] Testing Consistent Input (Concept A)...")
    d_consistent, _ = scanner.step(concept_A, concept_A)
    print(f"  Drift (Consistent): {d_consistent:.4f}")

    # 4. Test Novel/Inconsistent Input (Concept B)
    # Since B matches nothing in the sparse state, drift should be different.
    # Actually, for auto-associative, if B is orthogonal to A, 
    # Prediction = B @ State = B @ (A A^T) = (B.A) A^T ~ 0.
    # Reality = B.
    # Drift = ||B - 0|| = ||B|| = 1.0.
    # Whereas for A: ||A - A @ (A A^T)|| = ||A - A|| = 0.
    
    print("\n[Phase 3] Testing Novel Input (Concept B)...")
    d_novel, state = scanner.step(concept_B, concept_B)
    print(f"  Drift (Novel): {d_novel:.4f}")
    
    if d_novel > d_consistent:
        print("  ✅ Synaptic Drift Confirmed: Novel concepts surprise the brain.")
    else:
        print("  ❌ Synaptic Drift Check Failed.")

    # 5. Visualization
    from src.utils.visualization import visualize_synaptic_drift
    visualize_synaptic_drift(
        [{'state': state}], 
        [d_consistent, d_novel], 
        save_path="test_bdh_unit.png"
    )

if __name__ == "__main__":
    test_bdh_hebbian_logic()
