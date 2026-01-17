"""
Full Pipeline Evaluation Script
================================

Evaluates the complete system on real data and generates:
1. Accuracy, F1, Precision, Recall metrics
2. Per-signal contribution analysis
3. Confidence calibration statistics
4. Submission-ready CSV

Usage:
    # Full evaluation (loads real pipeline)
    python scripts/evaluate_full_pipeline.py --mode full
    
    # Generate submission
    python scripts/evaluate_full_pipeline.py --mode submit
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


@dataclass
class EvaluationResults:
    """Complete evaluation results."""
    accuracy: float
    f1_macro: float
    f1_weighted: float
    precision_macro: float
    recall_macro: float
    precision_consistent: float
    recall_consistent: float
    precision_contradict: float
    recall_contradict: float
    auc_roc: float
    confusion_matrix: List[List[int]]
    per_sample_results: List[Dict]
    signal_statistics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"{'Metric':<30} {'Value':>10}")
        print("-"*60)
        print(f"{'Accuracy':<30} {self.accuracy:>10.4f}")
        print(f"{'F1 (Macro)':<30} {self.f1_macro:>10.4f}")
        print(f"{'F1 (Weighted)':<30} {self.f1_weighted:>10.4f}")
        print(f"{'AUC-ROC':<30} {self.auc_roc:>10.4f}")
        print("-"*60)
        print(f"{'Precision (Consistent)':<30} {self.precision_consistent:>10.4f}")
        print(f"{'Recall (Consistent)':<30} {self.recall_consistent:>10.4f}")
        print(f"{'Precision (Contradict)':<30} {self.precision_contradict:>10.4f}")
        print(f"{'Recall (Contradict)':<30} {self.recall_contradict:>10.4f}")
        print("-"*60)
        print("\nConfusion Matrix:")
        print("              Pred:0  Pred:1")
        print(f"  Actual:0    {self.confusion_matrix[0][0]:>6}  {self.confusion_matrix[0][1]:>6}")
        print(f"  Actual:1    {self.confusion_matrix[1][0]:>6}  {self.confusion_matrix[1][1]:>6}")
        print("="*60)


def load_dataset(dataset_dir: str = "dataset") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Load train/test data and novel texts.
    
    Returns:
        train_df, test_df, novels dict
    """
    dataset_path = PROJECT_ROOT / dataset_dir
    
    # Load CSVs
    train_df = pd.read_csv(dataset_path / "train.csv")
    test_df = pd.read_csv(dataset_path / "test.csv")
    
    print(f"Loaded {len(train_df)} train samples, {len(test_df)} test samples")
    
    # Load novels
    novels = {}
    books_dir = dataset_path / "Books"
    if books_dir.exists():
        for txt_file in books_dir.glob("*.txt"):
            book_name = txt_file.stem
            with open(txt_file, 'r', encoding='utf-8') as f:
                novels[book_name] = f.read()
            print(f"  Loaded novel: {book_name} ({len(novels[book_name]):,} chars)")
    
    return train_df, test_df, novels


# Note: Mock evaluation removed - use full_evaluate_with_real_pipeline only


def full_evaluate_with_real_pipeline(
    train_df: pd.DataFrame, 
    novels: Dict[str, str]
) -> EvaluationResults:
    """
    Full evaluation using the real SOTA pipeline.
    
    This is slow but gives accurate metrics.
    """
    from training.classifier_head import FeatureExtractor, LinearClassifierHead, SignalFeatures
    import torch
    
    print("\n[Full Mode] Loading real pipeline...")
    
    # This will load heavy models
    extractor = FeatureExtractor(device="cuda")
    
    # Load trained classifier
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "classifier_head.pt"
    classifier = LinearClassifierHead()
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained classifier")
    
    classifier.eval()
    
    all_features = []
    all_labels = []
    all_probs = []
    
    from tqdm import tqdm
    
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Evaluating"):
        book_name = str(row.get('book_name', ''))
        backstory = str(row.get('content', ''))
        
        # Find matching novel
        novel_text = None
        for novel_name, text in novels.items():
            if novel_name.lower() in book_name.lower() or book_name.lower() in novel_name.lower():
                novel_text = text
                break
        
        if novel_text is None:
            print(f"  Warning: Novel not found for '{book_name}', skipping")
            continue
        
        # Get ground truth label
        label_str = str(row.get('label', '')).lower()
        if label_str == 'consistent':
            label = 1
        elif label_str == 'contradict':
            label = 0
        else:
            continue
        
        # Extract real features
        features = extractor.extract_features(
            novel_text=novel_text,
            backstory=backstory,
            sample_id=str(row.get('id', idx)),
            book_name=book_name
        )
        
        # Get classifier prediction
        x = features.to_tensor().unsqueeze(0)
        with torch.no_grad():
            prob = classifier(x).item()
        
        all_features.append(features)
        all_labels.append(label)
        all_probs.append(prob)
    
    all_preds = [1 if p > 0.5 else 0 for p in all_probs]
    
    return compute_metrics(all_labels, all_preds, all_probs, all_features)


def compute_metrics(
    labels: List[int],
    preds: List[int],
    probs: List[float],
    features: List[Any]
) -> EvaluationResults:
    """Compute all evaluation metrics."""
    
    labels_arr = np.array(labels)
    preds_arr = np.array(preds)
    probs_arr = np.array(probs)
    
    # Basic metrics
    accuracy = accuracy_score(labels_arr, preds_arr)
    f1_macro = f1_score(labels_arr, preds_arr, average='macro')
    f1_weighted = f1_score(labels_arr, preds_arr, average='weighted')
    precision_macro = precision_score(labels_arr, preds_arr, average='macro', zero_division=0)
    recall_macro = recall_score(labels_arr, preds_arr, average='macro', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(labels_arr, preds_arr, average=None, zero_division=0)
    recall_per_class = recall_score(labels_arr, preds_arr, average=None, zero_division=0)
    
    # AUC-ROC
    try:
        auc = roc_auc_score(labels_arr, probs_arr)
    except:
        auc = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds_arr).tolist()
    
    # Per-sample results
    per_sample = []
    for i, (label, pred, prob, feat) in enumerate(zip(labels, preds, probs, features)):
        per_sample.append({
            'sample_id': feat.sample_id if hasattr(feat, 'sample_id') else str(i),
            'true_label': label,
            'pred_label': pred,
            'probability': prob,
            'correct': label == pred,
        })
    
    # Signal statistics
    signal_stats = compute_signal_statistics(features, labels)
    
    return EvaluationResults(
        accuracy=accuracy,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        precision_consistent=float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        recall_consistent=float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        precision_contradict=float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        recall_contradict=float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        auc_roc=auc,
        confusion_matrix=cm,
        per_sample_results=per_sample,
        signal_statistics=signal_stats
    )


def compute_signal_statistics(features: List[Any], labels: List[int]) -> Dict[str, Dict[str, float]]:
    """Compute statistics for each signal grouped by label."""
    
    stats = {
        'nli_score': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'bdh_drift': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'evidence_score': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'temporal_violations': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
        'causal_violations': {'consistent_mean': 0, 'contradict_mean': 0, 'separation': 0},
    }
    
    # Group by label
    consistent_feats = [f for f, l in zip(features, labels) if l == 1]
    contradict_feats = [f for f, l in zip(features, labels) if l == 0]
    
    if not consistent_feats or not contradict_feats:
        return stats
    
    for signal in stats.keys():
        cons_vals = [getattr(f, signal, 0) for f in consistent_feats]
        cont_vals = [getattr(f, signal, 0) for f in contradict_feats]
        
        cons_mean = np.mean(cons_vals)
        cont_mean = np.mean(cont_vals)
        
        stats[signal]['consistent_mean'] = float(cons_mean)
        stats[signal]['contradict_mean'] = float(cont_mean)
        stats[signal]['separation'] = float(abs(cont_mean - cons_mean))
    
    return stats


def generate_submission(
    test_df: pd.DataFrame,
    novels: Dict[str, str],
    output_path: str = "results.csv",
    device: str = "cuda"
) -> pd.DataFrame:
    """
    Generate submission CSV for test set using real pipeline.
    """
    from training.classifier_head import FeatureExtractor, LinearClassifierHead
    import torch
    
    print(f"\nGenerating submission to {output_path}...")
    
    # Load classifier
    checkpoint_path = PROJECT_ROOT / "checkpoints" / "classifier_head.pt"
    classifier = LinearClassifierHead()
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained classifier")
    else:
        print("  Warning: No trained classifier found, using default weights")
    
    classifier.eval()
    
    # Use real feature extractor
    extractor = FeatureExtractor(device=device)
    
    results = []
    
    from tqdm import tqdm
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing"):
        sample_id = row.get('id', idx)
        book_name = str(row.get('book_name', ''))
        backstory = str(row.get('content', ''))
        
        # Find novel and extract real features
        novel_text = None
        for novel_name, text in novels.items():
            if novel_name.lower() in book_name.lower() or book_name.lower() in novel_name.lower():
                novel_text = text
                break
        
        if novel_text is None:
            print(f"  Warning: Novel not found for '{book_name}'")
            # Create default features for unknown novels
            from training.classifier_head import SignalFeatures
            features = SignalFeatures(
                sample_id=str(sample_id),
                book_name=book_name
            )
        else:
            features = extractor.extract_features(
                novel_text=novel_text,
                backstory=backstory,
                sample_id=str(sample_id),
                book_name=book_name
            )
        
        # Get prediction
        x = features.to_tensor().unsqueeze(0)
        with torch.no_grad():
            prob = classifier(x).item()
        
        pred = 1 if prob > 0.5 else 0
        confidence = prob if pred == 1 else (1 - prob)
        
        # Generate rationale based on signals
        if pred == 1:
            rationale = f"Backstory consistent (conf={confidence:.2f}): NLI={features.nli_score:.2f}, Drift={features.bdh_drift:.2f}"
        else:
            rationale = f"Backstory contradicts (conf={confidence:.2f}): NLI={features.nli_score:.2f}, Drift={features.bdh_drift:.2f}"
        
        results.append({
            'Story ID': sample_id,
            'Prediction': pred,
            'Rationale': rationale
        })
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    
    print(f"  Saved {len(results_df)} predictions to {output_path}")
    
    # Print distribution
    pred_counts = results_df['Prediction'].value_counts()
    print(f"  Prediction distribution:")
    print(f"    Consistent (1): {pred_counts.get(1, 0)}")
    print(f"    Contradict (0): {pred_counts.get(0, 0)}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Full pipeline evaluation")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "submit"],
        help="Evaluation mode: full (evaluate on train) or submit (generate submission)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.csv",
        help="Output path for submission CSV"
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default="evaluation_results.json",
        help="Path to save evaluation metrics"
    )
    
    args = parser.parse_args()
    
    # Load data
    train_df, test_df, novels = load_dataset()
    
    if args.mode == "full":
        # Full evaluation with real pipeline
        results = full_evaluate_with_real_pipeline(train_df, novels)
        results.print_summary()
        
        # Print signal statistics
        print("\nSignal Statistics:")
        print("-"*60)
        for signal, stats in results.signal_statistics.items():
            sep = stats['separation']
            print(f"  {signal:<25} Separation: {sep:.4f}")
        
        # Save results
        with open(args.save_metrics, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nMetrics saved to {args.save_metrics}")
        
    elif args.mode == "submit":
        # Generate submission using real pipeline
        generate_submission(
            test_df=test_df,
            novels=novels,
            output_path=args.output,
            device="cuda"
        )
        
        # Also run full eval on train to report metrics
        print("\nRunning evaluation on train set...")
        results = full_evaluate_with_real_pipeline(train_df, novels)
        results.print_summary()


if __name__ == "__main__":
    main()
