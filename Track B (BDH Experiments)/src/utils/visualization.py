"""
Synaptic Drift Visualization
===========================

Generates heatmaps of "Brain Activity" (Latent State Evolution) 
to demonstrate the BDH model's reaction to the narrative.

Includes:
- Static heatmaps
- Animated GIFs (for real-time drift visualization)
- Robust normalization

Uses: Matplotlib, Seaborn, PIL
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any
import io
from PIL import Image


# ============================================================================
# Utility Functions (from research/bdh_edufork)
# ============================================================================

def fig_to_pil_image(fig) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return image


def save_gif(images: List[Image.Image], save_path: str, duration: int = 500):
    """Save a list of PIL images as an animated GIF."""
    if not images:
        raise ValueError("Cannot save empty image list")

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        optimize=False
    )
    print(f"Animation saved to {save_path}")


def normalize_robust(arr: np.ndarray, percentile_low: float = 5, percentile_high: float = 95) -> np.ndarray:
    """Normalize array using robust percentile-based normalization."""
    if arr.size == 0:
        return arr
    low = np.percentile(arr, percentile_low)
    high = np.percentile(arr, percentile_high)
    if high - low > 1e-8:
        return np.clip((arr - low) / (high - low), 0, 1)
    return np.zeros_like(arr)


# ============================================================================
# Main Visualization Functions
# ============================================================================

def visualize_synaptic_drift(
    chunk_states: List[Dict[str, Any]],
    drift_scores: List[float] = None,
    save_path: str = "synaptic_drift.png",
    title: str = "BDH Synaptic Stability Scan"
):
    """
    Generate a heatmap of the latent state over time.
    
    Args:
        chunk_states: List of dicts with 'state' (tensor [hidden_dim])
        drift_scores: Optional list of drift scores to highlight anomalies
        save_path: Output path
    """
    # Extract states into matrix [Time, Hidden_Dim]
    # Handle both vector (Latent) and matrix (Synaptic) states
    processed_states = []
    for c in chunk_states:
        s = c['state']
        if isinstance(s, torch.Tensor):
            s = s.cpu().numpy()
        
        # If matrix [D, D], compute "Neuron Connectivity Strength" (Norm of rows)
        if s.ndim == 2:
            s = np.linalg.norm(s, axis=1) # [D] vector representing neuron 'importance'
            
        processed_states.append(s)

    matrix = np.stack(processed_states) # [T, H]
    
    # Normalize for visualization
    matrix = (matrix - matrix.mean()) / (matrix.std() + 1e-8)
    
    # Setup plot
    plt.figure(figsize=(12, 6))
    
    # Main Heatmap: Latent State Evolution
    ax1 = plt.subplot(2, 1, 1)
    sns.heatmap(matrix.T, cmap="viridis", cbar=True, ax=ax1)
    ax1.set_title("Neuron Activity (Latent State)")
    ax1.set_xlabel("Narrative Time (Chunks)")
    ax1.set_ylabel("Memory Dimensions")
    
    # Drift Score Plot
    if drift_scores:
        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        x = range(len(drift_scores))
        ax2.plot(x, drift_scores, color='red', linewidth=2)
        ax2.fill_between(x, drift_scores, color='red', alpha=0.3)
        ax2.set_title("Synaptic Drift (Anomaly Detection)")
        ax2.set_xlabel("Narrative Time (Chunks)")
        ax2.set_ylabel("Drift Magnitude")
        ax2.grid(True, alpha=0.3)
        
        # Add threshold line
        threshold = np.mean(drift_scores) + 2 * np.std(drift_scores)
        ax2.axhline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax2.legend()
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Visualization saved to {save_path}")

def generate_demo_visualization():
    """Generates a dummy visualization for the report/demo."""
    print("Generating DEMO visualization...")
    
    # Simulate 100 chunks of narrative
    T = 100
    Hidden = 50
    
    # Base state (stable story)
    base_state = np.random.randn(Hidden)
    
    chunk_states = []
    drift_scores = []
    
    current_state = base_state.copy()
    
    for t in range(T):
        # Drift: Story changes slightly
        noise = np.random.randn(Hidden) * 0.1
        
        # Sudden contradiction at t=60
        if 58 <= t <= 62:
            noise += np.random.randn(Hidden) * 2.0 # Huge spike
            
        current_state = 0.9 * current_state + noise
        
        chunk_states.append({'state': current_state.copy()})
        
        # Drift score = magnitude of change
        drift = np.linalg.norm(noise)
        drift_scores.append(drift)
        
    visualize_synaptic_drift(
        chunk_states, 
        drift_scores, 
        save_path="demo_synaptic_drift.png"
    )

if __name__ == "__main__":
    generate_demo_visualization()
