"""
Evaluation System
Classification metrics, BDH representation analysis, and ablation studies.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], save_path: Optional[str] = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Inconsistent', 'Consistent'],
                yticklabels=['Inconsistent', 'Consistent'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_bdh_representations(representations: List[torch.Tensor]) -> Dict:
    """
    Analyze BDH representation quality.
    
    Args:
        representations: List of representation tensors
        
    Returns:
        Analysis dictionary
    """
    stacked = torch.stack(representations)  # [num_samples, hidden_size]
    
    # Compute statistics
    mean_repr = stacked.mean(dim=0)
    std_repr = stacked.std(dim=0)
    
    # Sparsity (percentage of near-zero values)
    threshold = 0.01
    sparsity = (stacked.abs() < threshold).float().mean().item()
    
    # Diversity (average pairwise cosine distance)
    normalized = torch.nn.functional.normalize(stacked, p=2, dim=1)
    pairwise_sim = torch.mm(normalized, normalized.t())
    diversity = 1.0 - pairwise_sim.mean().item()
    
    return {
        'mean_activation': mean_repr.mean().item(),
        'std_activation': std_repr.std().item(),
        'sparsity': sparsity,
        'diversity': diversity,
        'num_samples': len(representations)
    }


def compare_representations(repr1: List[torch.Tensor], 
                          repr2: List[torch.Tensor],
                          name1: str = "Model 1",
                          name2: str = "Model 2") -> Dict:
    """
    Compare two sets of representations.
    
    Args:
        repr1: First set of representations
        repr2: Second set of representations
        name1: Name for first model
        name2: Name for second model
        
    Returns:
        Comparison dictionary
    """
    analysis1 = analyze_bdh_representations(repr1)
    analysis2 = analyze_bdh_representations(repr2)
    
    # Compute similarity between sets
    stacked1 = torch.stack(repr1)
    stacked2 = torch.stack(repr2)
    
    # Normalize
    norm1 = torch.nn.functional.normalize(stacked1, p=2, dim=1)
    norm2 = torch.nn.functional.normalize(stacked2, p=2, dim=1)
    
    # Average similarity
    if stacked1.size(0) == stacked2.size(0):
        similarity = (norm1 * norm2).sum(dim=1).mean().item()
    else:
        # Use minimum size
        min_size = min(stacked1.size(0), stacked2.size(0))
        similarity = (norm1[:min_size] * norm2[:min_size]).sum(dim=1).mean().item()
    
    return {
        name1: analysis1,
        name2: analysis2,
        'similarity': similarity
    }


def ablation_study_results(results: Dict[str, Dict]) -> Dict:
    """
    Analyze ablation study results.
    
    Args:
        results: Dictionary mapping configuration names to metric dictionaries
        
    Returns:
        Analysis of ablation study
    """
    best_config = None
    best_f1 = 0.0
    
    for config_name, metrics in results.items():
        f1 = metrics.get('f1_score', 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_config = config_name
    
    return {
        'best_configuration': best_config,
        'best_f1_score': best_f1,
        'all_results': results
    }


def generate_evaluation_report(y_true: List[int],
                              y_pred: List[int],
                              y_probs: Optional[List[float]] = None,
                              save_path: Optional[str] = None) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Optional prediction probabilities
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    metrics = compute_classification_metrics(y_true, y_pred)
    
    report = "=" * 60 + "\n"
    report += "EVALUATION REPORT\n"
    report += "=" * 60 + "\n\n"
    
    report += "Classification Metrics:\n"
    report += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
    report += f"  Precision: {metrics['precision']:.4f}\n"
    report += f"  Recall:    {metrics['recall']:.4f}\n"
    report += f"  F1 Score:  {metrics['f1_score']:.4f}\n\n"
    
    # Per-class breakdown
    cm = confusion_matrix(y_true, y_pred)
    report += "Confusion Matrix:\n"
    report += f"  True Negatives (Inconsistent):  {cm[0, 0]}\n"
    report += f"  False Positives:                {cm[0, 1]}\n"
    report += f"  False Negatives:                {cm[1, 0]}\n"
    report += f"  True Positives (Consistent):    {cm[1, 1]}\n\n"
    
    if y_probs is not None:
        # Probability statistics
        probs_array = np.array(y_probs)
        report += "Prediction Confidence:\n"
        report += f"  Mean Probability: {probs_array.mean():.4f}\n"
        report += f"  Std Probability:  {probs_array.std():.4f}\n"
        report += f"  Min Probability:  {probs_array.min():.4f}\n"
        report += f"  Max Probability:  {probs_array.max():.4f}\n\n"
    
    report += "=" * 60 + "\n"
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
    
    return report

