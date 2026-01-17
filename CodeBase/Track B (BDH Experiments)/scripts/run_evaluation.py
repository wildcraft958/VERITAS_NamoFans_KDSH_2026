#!/usr/bin/env python3
"""
Complete Evaluation Pipeline for KDSH 2026
==========================================

Runs the full BDH-based narrative consistency pipeline on the dataset,
generates visualizations, and produces submission-ready results.

Features:
- Full pipeline evaluation on train/test data
- Synaptic drift visualizations per sample
- Per-signal contribution analysis
- Detailed evaluation report
- Submission CSV generation

Usage:
    # Full evaluation with visualizations
    python scripts/run_evaluation.py --mode full --visualize
    
    # Quick evaluation (subset)
    python scripts/run_evaluation.py --mode quick --samples 10
    
    # Generate submission only
    python scripts/run_evaluation.py --mode submit
    
    # CPU-only mode
    python scripts/run_evaluation.py --mode full --device cpu

Requirements:
    - GPU: 4-5 GB VRAM (recommended)
    - CPU: 16-20 GB RAM (if no GPU)
    - ~10-30 minutes for full evaluation on GPU
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path (early, before dependency check)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Dependency Installation
# =============================================================================

REQUIRED_PACKAGES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'torch': 'torch',
    'tqdm': 'tqdm',
    'sklearn': 'scikit-learn',
    'transformers': 'transformers',
    'sentence_transformers': 'sentence-transformers',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
}

def check_and_install_dependencies():
    """Check for required packages and install if missing."""
    missing = []
    
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    
    if missing:
        print(f"📦 Installing missing dependencies: {', '.join(missing)}")
        print("=" * 60)
        
        for package in missing:
            try:
                print(f"   Installing {package}...")
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package, "-q"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"   ✅ {package} installed")
            except subprocess.CalledProcessError:
                print(f"   ❌ Failed to install {package}")
                print(f"      Run: pip install {package}")
        
        print("=" * 60)
        print("🔄 Re-importing dependencies...")
        print()

# Run dependency check on import
check_and_install_dependencies()

# Now import everything
import json
import argparse
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_id: str
    book_name: str
    true_label: int
    pred_label: int
    confidence: float
    
    # Signal scores
    nli_score: float
    bdh_drift: float
    evidence_score: float
    temporal_violations: int
    causal_violations: int
    
    # Evidence
    rationale: str
    supporting_evidence: List[str]
    contradicting_evidence: List[str]
    
    # Timing
    processing_time: float


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    # Metadata
    timestamp: str
    device: str
    mode: str
    num_samples: int
    
    # Overall metrics
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_consistent: float
    recall_consistent: float
    precision_contradict: float
    recall_contradict: float
    auc_roc: float
    
    # Confusion matrix
    confusion_matrix: List[List[int]]
    
    # Signal statistics
    signal_stats: Dict[str, Dict[str, float]]
    
    # Per-sample results
    sample_results: List[Dict]
    
    # Timing
    total_time: float
    avg_time_per_sample: float
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"📊 Report saved to {path}")


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_sample_drift(
    chunk_states: List[np.ndarray],
    drift_scores: List[float],
    sample_id: str,
    book_name: str,
    prediction: str,
    output_dir: Path
):
    """Generate synaptic drift visualization for a single sample."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if not chunk_states or not drift_scores:
        return None
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'medium',
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'grid.color': '#e9ecef',
        'grid.linewidth': 0.8,
    })
    
    # Convert to matrix
    matrix = np.stack(chunk_states)
    if matrix.ndim == 3:  # [T, D, D] -> compute row norms
        matrix = np.linalg.norm(matrix, axis=2)
    
    # Normalize
    matrix = (matrix - matrix.mean()) / (matrix.std() + 1e-8)
    
    # Create figure with better spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), 
                                    gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
    
    # Color scheme
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Heatmap with improved styling
    hm = sns.heatmap(matrix.T[:50, :], cmap="viridis", cbar=True, ax=ax1,
                     cbar_kws={'label': 'Activation', 'shrink': 0.8})
    ax1.set_title(f"Neuron Activity (Latent State)", fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylabel("Memory Dimensions", fontsize=12, fontweight='medium')
    ax1.set_xlabel("Narrative Time (Chunks)", fontsize=12, fontweight='medium')
    
    # Drift scores with improved styling
    x = range(len(drift_scores))
    ax2.plot(x, drift_scores, color='#e74c3c', linewidth=2.5, zorder=5)
    ax2.fill_between(x, drift_scores, color='#e74c3c', alpha=0.25, zorder=4)
    
    # Prediction color coding
    pred_color = '#27ae60' if prediction.lower() == 'consistent' else '#e74c3c'
    ax2.set_title(f"Synaptic Drift (Anomaly Detection)", 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel("Narrative Time (Chunks)", fontsize=12, fontweight='medium')
    ax2.set_ylabel("Drift Magnitude", fontsize=12, fontweight='medium')
    ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Threshold line with better styling
    if drift_scores:
        threshold = np.mean(drift_scores) + 2 * np.std(drift_scores)
        ax2.axhline(threshold, color='#2c3e50', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.2f})', zorder=6)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, 
                  shadow=True, fontsize=11)
    
    # Add prediction badge
    ax2.text(0.02, 0.95, f"Prediction: {prediction}", transform=ax2.transAxes,
             fontsize=12, fontweight='bold', color='white',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=pred_color, alpha=0.9),
             verticalalignment='top')
    
    plt.tight_layout(pad=2.0)
    
    # Save with high quality
    output_path = output_dir / f"drift_{sample_id}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return str(output_path)


def visualize_signal_contributions(report: EvaluationReport, output_dir: Path):
    """Generate signal contribution analysis visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
    })
    
    stats = report.signal_stats
    signals = list(stats.keys())
    
    # Color palette
    colors = {'consistent': '#27ae60', 'contradict': '#e74c3c'}
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('Signal Contribution Analysis', fontsize=18, fontweight='bold', y=1.02)
    axes = axes.flatten()
    
    for i, signal in enumerate(signals[:5]):
        ax = axes[i]
        cons_mean = stats[signal].get('consistent_mean', 0)
        cont_mean = stats[signal].get('contradict_mean', 0)
        
        bars = ax.bar(['Consistent', 'Contradict'], [cons_mean, cont_mean],
                     color=[colors['consistent'], colors['contradict']], 
                     alpha=0.85, edgecolor='white', linewidth=2)
        ax.set_title(f"{signal.replace('_', ' ').title()}", fontsize=13, fontweight='bold')
        ax.set_ylabel("Mean Score", fontsize=11)
        ax.set_ylim(0, max(cons_mean, cont_mean) * 1.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels with better styling
        for bar, val in zip(bars, [cons_mean, cont_mean]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Summary in last subplot with styled metrics box
    ax = axes[5]
    ax.set_facecolor('#f8f9fa')
    
    # Create metrics display
    metrics_text = [
        (0.5, 0.75, f"Accuracy", f"{report.accuracy:.1%}", '#3498db'),
        (0.5, 0.50, f"F1 Macro", f"{report.f1_macro:.1%}", '#9b59b6'),
        (0.5, 0.25, f"AUC-ROC", f"{report.auc_roc:.3f}", '#e67e22'),
    ]
    
    for x, y, label, value, color in metrics_text:
        ax.text(x, y + 0.05, label, transform=ax.transAxes, 
               fontsize=12, ha='center', color='#7f8c8d', fontweight='medium')
        ax.text(x, y - 0.05, value, transform=ax.transAxes, 
               fontsize=18, ha='center', color=color, fontweight='bold')
    
    ax.axis('off')
    ax.set_title("Overall Metrics", fontsize=13, fontweight='bold')
    
    plt.tight_layout(pad=2.5)
    
    output_path = output_dir / "signal_contributions.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📈 Signal analysis saved to {output_path}")
    return str(output_path)


def visualize_confusion_matrix(report: EvaluationReport, output_dir: Path):
    """Generate confusion matrix visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    cm = np.array(report.confusion_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Custom colormap
    cmap = sns.light_palette("#3498db", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                annot_kws={'size': 20, 'weight': 'bold'},
                linewidths=3, linecolor='white',
                square=True, cbar_kws={'shrink': 0.8},
                xticklabels=['Contradict', 'Consistent'],
                yticklabels=['Contradict', 'Consistent'])
    
    ax.set_title(f"Confusion Matrix", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Predicted", fontsize=13, fontweight='medium', labelpad=10)
    ax.set_ylabel("Actual", fontsize=13, fontweight='medium', labelpad=10)
    
    # Add accuracy badge
    ax.text(1.15, 0.5, f"Accuracy\n{report.accuracy:.1%}", 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#27ae60', alpha=0.9),
            color='white')
    
    plt.tight_layout(pad=2.0)
    
    output_path = output_dir / "confusion_matrix.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"🔢 Confusion matrix saved to {output_path}")
    return str(output_path)


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def load_dataset(dataset_dir: str = "dataset") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Load train/test data and novel texts."""
    dataset_path = PROJECT_ROOT / dataset_dir
    
    train_df = pd.read_csv(dataset_path / "train.csv")
    test_df = pd.read_csv(dataset_path / "test.csv")
    
    print(f"📚 Loaded {len(train_df)} train samples, {len(test_df)} test samples")
    
    # Load novels
    novels = {}
    books_dir = dataset_path / "Books"
    if books_dir.exists():
        for txt_file in books_dir.glob("*.txt"):
            book_name = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as f:
                novels[book_name] = f.read()
            print(f"   📖 {book_name}: {len(novels[book_name]):,} chars")
    
    return train_df, test_df, novels


def get_book_for_sample(book_name: str, novels: Dict[str, str]) -> Optional[str]:
    """Find matching novel text for a sample."""
    # Direct match
    if book_name in novels:
        return novels[book_name]
    
    # Fuzzy match
    book_lower = book_name.lower()
    for novel_name, text in novels.items():
        if novel_name.lower() in book_lower or book_lower in novel_name.lower():
            return text
    
    return None


def run_evaluation(
    df: pd.DataFrame,
    novels: Dict[str, str],
    pipeline,
    device: str = "cuda",
    visualize: bool = False,
    output_dir: Path = None,
    max_samples: int = None
) -> EvaluationReport:
    """Run evaluation on dataset."""
    
    if output_dir and visualize:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Limit samples if requested
    eval_df = df.head(max_samples) if max_samples else df
    
    start_time = time.time()
    
    print(f"\n🔬 Evaluating {len(eval_df)} samples...")
    print("=" * 60)
    
    # Track current book for efficient processing
    current_book = None
    
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Processing"):
        sample_start = time.time()
        
        sample_id = str(row.get('id', idx))
        book_name = str(row.get('book_name', ''))
        backstory = str(row.get('content', ''))
        
        # Get label
        label_str = str(row.get('label', '')).lower()
        if label_str == 'consistent':
            true_label = 1
        elif label_str == 'contradict':
            true_label = 0
        else:
            continue  # Skip unknown labels
        
        # Get novel text
        novel_text = get_book_for_sample(book_name, novels)
        if novel_text is None:
            print(f"⚠️  Novel not found for '{book_name}', skipping")
            continue
        
        # Ingest novel if changed
        if book_name != current_book:
            print(f"\n📖 Ingesting: {book_name}")
            pipeline.ingest_novel(novel_text)
            current_book = book_name
        
        # Get verdict
        try:
            verdict = pipeline.check_consistency(backstory)
            
            pred_label = 1 if verdict.consistent else 0
            confidence = verdict.confidence
            
            # Extract signal scores
            nli_score = verdict.nli_score
            bdh_drift = verdict.perplexity_score
            evidence_score = len(verdict.supporting_evidence) / max(1, len(verdict.supporting_evidence) + len(verdict.contradicting_evidence))
            temporal_violations = verdict.temporal_violations
            causal_violations = verdict.causal_violations
            
            rationale = verdict.rationale
            supporting = [e.text if hasattr(e, 'text') else str(e) for e in verdict.supporting_evidence[:3]]
            contradicting = [e.text if hasattr(e, 'text') else str(e) for e in verdict.contradicting_evidence[:3]]
            
        except Exception as e:
            print(f"⚠️  Error processing sample {sample_id}: {e}")
            pred_label = 0
            confidence = 0.5
            nli_score = bdh_drift = evidence_score = 0.0
            temporal_violations = causal_violations = 0
            rationale = f"Error: {str(e)}"
            supporting = []
            contradicting = []
        
        processing_time = time.time() - sample_start
        
        # Store result
        result = SampleResult(
            sample_id=sample_id,
            book_name=book_name,
            true_label=true_label,
            pred_label=pred_label,
            confidence=confidence,
            nli_score=nli_score,
            bdh_drift=bdh_drift,
            evidence_score=evidence_score,
            temporal_violations=temporal_violations,
            causal_violations=causal_violations,
            rationale=rationale,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            processing_time=processing_time
        )
        results.append(result)
        
        all_labels.append(true_label)
        all_preds.append(pred_label)
        all_probs.append(confidence if pred_label == 1 else 1 - confidence)
        
        # Generate visualization if requested
        if visualize and output_dir and hasattr(pipeline, 'bdh') and hasattr(pipeline.bdh, 'history'):
            try:
                chunk_states = [h.get('state', np.zeros(50)) for h in pipeline.bdh.history[-100:]]
                drift_scores = [h.get('drift', 0) for h in pipeline.bdh.history[-100:]]
                pred_str = "Consistent" if pred_label == 1 else "Contradict"
                visualize_sample_drift(chunk_states, drift_scores, sample_id, book_name, pred_str, vis_dir)
            except:
                pass
    
    total_time = time.time() - start_time
    
    # Compute metrics
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)
    
    accuracy = accuracy_score(labels_arr, preds_arr)
    f1_macro = f1_score(labels_arr, preds_arr, average='macro')
    f1_weighted = f1_score(labels_arr, preds_arr, average='weighted')
    
    precision_per_class = precision_score(labels_arr, preds_arr, average=None, zero_division=0)
    recall_per_class = recall_score(labels_arr, preds_arr, average=None, zero_division=0)
    
    try:
        auc = roc_auc_score(labels_arr, probs_arr)
    except:
        auc = 0.5
    
    cm = confusion_matrix(labels_arr, preds_arr).tolist()
    
    # Signal statistics
    signal_stats = compute_signal_statistics(results)
    
    # Create report
    report = EvaluationReport(
        timestamp=datetime.now().isoformat(),
        device=device,
        mode="evaluation",
        num_samples=len(results),
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision_consistent=float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        recall_consistent=float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        precision_contradict=float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        recall_contradict=float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        auc_roc=auc,
        confusion_matrix=cm,
        signal_stats=signal_stats,
        sample_results=[asdict(r) for r in results],
        total_time=total_time,
        avg_time_per_sample=total_time / max(1, len(results))
    )
    
    return report


def compute_signal_statistics(results: List[SampleResult]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each signal grouped by label."""
    stats = {
        'nli_score': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'bdh_drift': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'evidence_score': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'temporal_violations': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'causal_violations': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
    }
    
    consistent = [r for r in results if r.true_label == 1]
    contradict = [r for r in results if r.true_label == 0]
    
    if not consistent or not contradict:
        return stats
    
    for signal in stats.keys():
        cons_vals = [getattr(r, signal, 0) for r in consistent]
        cont_vals = [getattr(r, signal, 0) for r in contradict]
        
        cons_mean = np.mean(cons_vals)
        cont_mean = np.mean(cont_vals)
        
        stats[signal]['consistent_mean'] = float(cons_mean)
        stats[signal]['contradict_mean'] = float(cont_mean)
        stats[signal]['separation'] = float(abs(cont_mean - cons_mean))
    
    return stats


def generate_submission(
    test_df: pd.DataFrame,
    novels: Dict[str, str],
    pipeline,
    output_path: str = "results.csv"
) -> pd.DataFrame:
    """Generate submission CSV for test set."""
    
    print(f"\n📝 Generating submission to {output_path}...")
    
    results = []
    current_book = None
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Generating"):
        sample_id = row.get('id', idx)
        book_name = str(row.get('book_name', ''))
        backstory = str(row.get('content', ''))
        
        # Get novel
        novel_text = get_book_for_sample(book_name, novels)
        
        if novel_text and book_name != current_book:
            pipeline.ingest_novel(novel_text)
            current_book = book_name
        
        # Get prediction
        try:
            if novel_text:
                verdict = pipeline.check_consistency(backstory)
                pred = 1 if verdict.consistent else 0
                confidence = verdict.confidence
                rationale = verdict.rationale
            else:
                pred = 1  # Default to consistent if novel not found
                confidence = 0.5
                rationale = "Novel not found, defaulting to consistent"
        except Exception as e:
            pred = 1
            confidence = 0.5
            rationale = f"Error: {str(e)}"
        
        results.append({
            'Story ID': sample_id,
            'Prediction': pred,
            'Rationale': rationale[:500]  # Limit rationale length
        })
    
    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    # Print distribution
    pred_counts = results_df['Prediction'].value_counts()
    print(f"\n✅ Saved {len(results_df)} predictions to {output_path}")
    print(f"   Consistent (1): {pred_counts.get(1, 0)}")
    print(f"   Contradict (0): {pred_counts.get(0, 0)}")
    
    return results_df


def print_summary(report: EvaluationReport):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 60)
    print(f"{'Samples Evaluated':<30} {report.num_samples:>15}")
    print(f"{'Total Time':<30} {report.total_time:>12.1f}s")
    print(f"{'Avg Time/Sample':<30} {report.avg_time_per_sample:>12.2f}s")
    print("-" * 60)
    print(f"{'Accuracy':<30} {report.accuracy:>14.2%}")
    print(f"{'F1 (Macro)':<30} {report.f1_macro:>14.2%}")
    print(f"{'F1 (Weighted)':<30} {report.f1_weighted:>14.2%}")
    print(f"{'AUC-ROC':<30} {report.auc_roc:>15.4f}")
    print("-" * 60)
    print(f"{'Precision (Consistent)':<30} {report.precision_consistent:>14.2%}")
    print(f"{'Recall (Consistent)':<30} {report.recall_consistent:>14.2%}")
    print(f"{'Precision (Contradict)':<30} {report.precision_contradict:>14.2%}")
    print(f"{'Recall (Contradict)':<30} {report.recall_contradict:>14.2%}")
    print("-" * 60)
    print("\n📉 Confusion Matrix:")
    print("              Pred:0  Pred:1")
    print(f"  Actual:0    {report.confusion_matrix[0][0]:>6}  {report.confusion_matrix[0][1]:>6}")
    print(f"  Actual:1    {report.confusion_matrix[1][0]:>6}  {report.confusion_matrix[1][1]:>6}")
    print("=" * 60)
    
    print("\n📊 Signal Separation (Higher = Better Discriminator):")
    print("-" * 60)
    for signal, stats in sorted(report.signal_stats.items(), key=lambda x: -x[1]['separation']):
        sep = stats['separation']
        print(f"  {signal:<25} Separation: {sep:>8.4f}")
    print("=" * 60)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Complete BDH Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full evaluation with visualizations
    python scripts/run_evaluation.py --mode full --visualize
    
    # Quick test (10 samples)
    python scripts/run_evaluation.py --mode quick --samples 10
    
    # Generate submission only
    python scripts/run_evaluation.py --mode submit
    
    # CPU-only mode
    python scripts/run_evaluation.py --mode full --device cpu
        """
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "quick", "submit"],
        help="Evaluation mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Max samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--submission-file",
        type=str,
        default="results.csv",
        help="Output path for submission CSV"
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("🧠 BDH Narrative Consistency Evaluation")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = "cpu"
    
    if args.device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load dataset
    train_df, test_df, novels = load_dataset()
    
    # Load pipeline
    print("\n🔧 Loading SOTA Pipeline...")
    from src.pipeline import create_sota_pipeline
    pipeline = create_sota_pipeline(device=args.device)
    print("✅ Pipeline loaded successfully")
    
    if args.mode == "quick":
        # Quick evaluation on subset
        max_samples = args.samples or 10
        print(f"\n🚀 Quick mode: evaluating {max_samples} samples")
        
        report = run_evaluation(
            df=train_df,
            novels=novels,
            pipeline=pipeline,
            device=args.device,
            visualize=args.visualize,
            output_dir=output_dir,
            max_samples=max_samples
        )
        
        print_summary(report)
        report.save(output_dir / "evaluation_report.json")
        
    elif args.mode == "full":
        # Full evaluation on training set
        print("\n🔬 Full evaluation on training set...")
        
        report = run_evaluation(
            df=train_df,
            novels=novels,
            pipeline=pipeline,
            device=args.device,
            visualize=args.visualize,
            output_dir=output_dir,
            max_samples=args.samples
        )
        
        print_summary(report)
        report.save(output_dir / "evaluation_report.json")
        
        # Generate visualizations
        if args.visualize:
            print("\n🎨 Generating visualizations...")
            visualize_signal_contributions(report, output_dir)
            visualize_confusion_matrix(report, output_dir)
        
    elif args.mode == "submit":
        # Generate submission
        generate_submission(
            test_df=test_df,
            novels=novels,
            pipeline=pipeline,
            output_path=str(PROJECT_ROOT / args.submission_file)
        )
        
        # Also run evaluation on train for metrics
        print("\n📊 Evaluating on train set for metrics...")
        report = run_evaluation(
            df=train_df,
            novels=novels,
            pipeline=pipeline,
            device=args.device,
            visualize=False,
            output_dir=output_dir
        )
        print_summary(report)
        report.save(output_dir / "evaluation_report.json")
    
    print("\n✅ Evaluation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
