"""
Classifier Head for BDH Multi-Signal Fusion
=============================================

Trains a lightweight classifier on top of the 5 signal outputs:
1. NLI Score (DeBERTa contradiction probability)
2. BDH Synaptic Drift (Hebbian state surprise)
3. Evidence Score (Cross-encoder passage matching)
4. Temporal Score (Allen's Algebra violations)
5. Causal Score (Constraint graph violations)

This replaces hand-tuned signal weights with learned weights,
enabling task-specific adaptation on synthetic/real data.

Usage:
    # Training
    python training/classifier_head.py --mode train --epochs 50
    
    # Evaluation
    python training/classifier_head.py --mode eval
    
    # Export weights for pipeline
    python training/classifier_head.py --mode export
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class SignalFeatures:
    """Features extracted from multi-signal pipeline."""
    sample_id: str
    book_name: str
    
    # Raw signal scores (5 signals)
    nli_score: float = 0.0           # Higher = more contradictory
    bdh_drift: float = 0.0           # Higher = more surprise/inconsistency
    evidence_score: float = 0.0      # Higher = better evidence match
    temporal_violations: int = 0     # Count of timeline violations
    causal_violations: int = 0       # Count of causal violations
    
    # Derived features
    nli_confidence: float = 0.0      # Model confidence in NLI prediction
    contradiction_evidence_count: int = 0
    supporting_evidence_count: int = 0
    
    # Label
    label: int = -1  # 1 = consistent, 0 = contradict
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to feature tensor for classifier."""
        return torch.tensor([
            self.nli_score,
            self.bdh_drift,
            self.evidence_score,
            float(self.temporal_violations),
            float(self.causal_violations),
            self.nli_confidence,
            float(self.contradiction_evidence_count),
            float(self.supporting_evidence_count),
        ], dtype=torch.float32)
    
    @property
    def num_features(self) -> int:
        return 8
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SignalDataset(Dataset):
    """PyTorch Dataset for signal features."""
    
    def __init__(self, features: List[SignalFeatures]):
        self.features = features
        self.num_features = features[0].num_features if features else 8
        
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.features[idx]
        x = feat.to_tensor()
        y = torch.tensor(feat.label, dtype=torch.float32)
        return x, y


# =============================================================================
# Classifier Models
# =============================================================================

class LinearClassifierHead(nn.Module):
    """
    Simple linear classifier - learns optimal signal weights.
    
    Equivalent to learning the weighted sum:
        score = w1*nli + w2*drift + w3*evidence + w4*temporal + w5*causal + bias
    
    Interpretable: weights directly show feature importance.
    """
    
    def __init__(self, num_features: int = 8):
        super().__init__()
        self.fc = nn.Linear(num_features, 1)
        
        # Initialize with reasonable priors (similar to hand-tuned weights)
        with torch.no_grad():
            # NLI and drift should start negative (higher = more contradiction)
            # Evidence should start positive (higher = more support)
            self.fc.weight[0, 0] = -0.35  # nli_score
            self.fc.weight[0, 1] = -0.25  # bdh_drift
            self.fc.weight[0, 2] = 0.20   # evidence_score
            self.fc.weight[0, 3] = -0.10  # temporal_violations
            self.fc.weight[0, 4] = -0.10  # causal_violations
            self.fc.weight[0, 5] = 0.0    # nli_confidence
            self.fc.weight[0, 6] = -0.05  # contradiction_evidence_count
            self.fc.weight[0, 7] = 0.05   # supporting_evidence_count
            self.fc.bias[0] = 0.5         # Start neutral
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_features] signal features
            
        Returns:
            [batch, 1] consistency probability
        """
        return torch.sigmoid(self.fc(x))
    
    def get_weights(self) -> Dict[str, float]:
        """Get learned weights for interpretability."""
        weight_names = [
            'nli_score', 'bdh_drift', 'evidence_score',
            'temporal_violations', 'causal_violations',
            'nli_confidence', 'contradiction_count', 'supporting_count'
        ]
        weights = self.fc.weight[0].detach().cpu().numpy()
        bias = self.fc.bias[0].item()
        
        result = {name: float(w) for name, w in zip(weight_names, weights)}
        result['bias'] = bias
        return result


class MLPClassifierHead(nn.Module):
    """
    Two-layer MLP classifier - captures non-linear signal interactions.
    
    Architecture:
        Input(8) -> Linear(32) -> ReLU -> Dropout -> Linear(1) -> Sigmoid
    
    Can learn interactions like:
        "High NLI + Low drift = strong contradiction"
        "High evidence + some violations = uncertain"
    """
    
    def __init__(self, num_features: int = 8, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionClassifierHead(nn.Module):
    """
    Attention-based classifier - learns which signals to attend to.
    
    Uses self-attention to weight signals dynamically based on input.
    Useful when signal importance varies by sample.
    """
    
    def __init__(self, num_features: int = 8, num_heads: int = 2):
        super().__init__()
        self.num_features = num_features
        
        # Project each feature to a query/key/value
        self.feature_embed = nn.Linear(1, 16)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=16, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.Linear(16 * num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        batch_size = x.size(0)
        
        # Reshape to [batch, num_features, 1] and embed
        x = x.unsqueeze(-1)  # [batch, num_features, 1]
        x = self.feature_embed(x)  # [batch, num_features, 16]
        
        # Self-attention over features
        attn_out, _ = self.attention(x, x, x)  # [batch, num_features, 16]
        
        # Flatten and project
        flat = attn_out.reshape(batch_size, -1)  # [batch, num_features * 16]
        return self.output(flat)


# =============================================================================
# Feature Extraction
# =============================================================================

class FeatureExtractor:
    """
    Extract signal features from the SOTA pipeline.
    
    Runs each component and collects scores for classifier training.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._pipeline = None
        
    def _load_pipeline(self):
        """Lazy-load the full pipeline (heavy)."""
        if self._pipeline is None:
            print("Loading SOTA Pipeline for feature extraction...")
            from src.pipeline.sota_pipeline import SOTANarrativeConsistencyPipeline
            self._pipeline = SOTANarrativeConsistencyPipeline(device=str(self.device))
        return self._pipeline
    
    def extract_features(
        self, 
        novel_text: str, 
        backstory: str,
        sample_id: str = "",
        book_name: str = ""
    ) -> SignalFeatures:
        """
        Extract all signal features for a single sample.
        
        Args:
            novel_text: Full novel text
            backstory: Character backstory to verify
            sample_id: Unique identifier
            book_name: Name of the novel
            
        Returns:
            SignalFeatures with all extracted scores
        """
        pipeline = self._load_pipeline()
        
        # Ingest novel if needed
        if pipeline.novel_text != novel_text:
            pipeline.ingest_novel(novel_text)
        
        # Get full verdict
        verdict = pipeline.verify_backstory(backstory)
        
        # Calculate evidence ratio (supporting / total evidence)
        supporting_count = len(verdict.supporting_evidence)
        contradicting_count = len(verdict.contradicting_evidence)
        total_evidence = supporting_count + contradicting_count
        evidence_ratio = supporting_count / max(1, total_evidence)
        
        # Extract features from verdict
        features = SignalFeatures(
            sample_id=sample_id,
            book_name=book_name,
            nli_score=verdict.nli_score,
            bdh_drift=verdict.perplexity_score,
            evidence_score=evidence_ratio,  # Ratio of supporting evidence
            temporal_violations=verdict.temporal_violations,
            causal_violations=verdict.causal_violations,
            nli_confidence=verdict.confidence,
            contradiction_evidence_count=contradicting_count,
            supporting_evidence_count=supporting_count,
        )
        
        return features
    
    def extract_batch(
        self,
        samples: List[Dict],
        novels: Dict[str, str],
        progress: bool = True
    ) -> List[SignalFeatures]:
        """
        Extract features for multiple samples.
        
        Args:
            samples: List of dicts with 'id', 'book_name', 'content', 'label'
            novels: Dict mapping book_name to novel text
            progress: Show progress bar
            
        Returns:
            List of SignalFeatures
        """
        from tqdm import tqdm
        
        features = []
        iterator = tqdm(samples, desc="Extracting features") if progress else samples
        
        for sample in iterator:
            book_name = sample['book_name']
            if book_name not in novels:
                print(f"Warning: Novel not found for {book_name}, skipping")
                continue
            
            feat = self.extract_features(
                novel_text=novels[book_name],
                backstory=sample['content'],
                sample_id=sample['id'],
                book_name=book_name
            )
            
            # Set label
            label_str = sample.get('label', '').lower()
            if label_str == 'consistent':
                feat.label = 1
            elif label_str == 'contradict':
                feat.label = 0
            else:
                feat.label = -1  # Unknown
            
            features.append(feat)
        
        return features





# =============================================================================
# Training Loop
# =============================================================================

class ClassifierTrainer:
    """
    Trainer for the classifier head.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        self.criterion = nn.BCELoss()
        
        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_accuracy = 0.0
        self.best_state = None
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(-1)
            
            self.optimizer.zero_grad()
            pred = self.model(batch_x)
            loss = self.criterion(pred, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device).unsqueeze(-1)
            
            probs = self.model(batch_x)
            loss = self.criterion(probs, batch_y)
            total_loss += loss.item()
            
            preds = (probs > 0.5).float()
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with early stopping.
        
        Returns:
            Training history and best metrics
        """
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        patience_counter = 0
        
        print(f"\nTraining Classifier Head")
        print(f"{'='*50}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"{'='*50}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            # Update scheduler
            self.scheduler.step()
            
            # Check for improvement
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                improved = "✓"
            else:
                patience_counter += 1
                improved = ""
            
            # Logging
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f} {improved}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        # Save model
        if save_path:
            self.save_model(save_path)
        
        # Final evaluation
        final_metrics = self.evaluate(val_loader)
        
        return {
            'best_accuracy': self.best_accuracy,
            'final_metrics': final_metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Model loaded from {path}")


# =============================================================================
# Main Functions
# =============================================================================

def prepare_data(
    dataset_path: str = "dataset",
    test_size: float = 0.2,
    random_state: int = 42,
    device: str = "cuda"
) -> Tuple[DataLoader, DataLoader, List[SignalFeatures]]:
    """
    Prepare training and validation data using real feature extraction.
    
    Args:
        dataset_path: Path to dataset directory
        test_size: Fraction for validation
        random_state: Random seed
        device: Device for model inference
        
    Returns:
        train_loader, val_loader, all_features
    """
    import pandas as pd
    from pathlib import Path
    
    dataset_dir = PROJECT_ROOT / dataset_path
    train_df = pd.read_csv(dataset_dir / "train.csv")
    
    # Load novels
    novels = {}
    books_dir = dataset_dir / "Books"
    for txt_file in books_dir.glob("*.txt"):
        book_name = txt_file.stem
        with open(txt_file, 'r', encoding='utf-8') as f:
            novels[book_name] = f.read()
    
    print(f"Loaded {len(train_df)} samples, {len(novels)} novels")
    
    # Extract real features
    extractor = FeatureExtractor(device=device)
    samples = train_df.to_dict('records')
    all_features = extractor.extract_batch(samples, novels, progress=True)
    
    # Split into train/val
    train_features, val_features = train_test_split(
        all_features, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[f.label for f in all_features]
    )
    
    print(f"Train samples: {len(train_features)}")
    print(f"Val samples: {len(val_features)}")
    
    # Create dataloaders
    train_dataset = SignalDataset(train_features)
    val_dataset = SignalDataset(val_features)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, all_features


def train_classifier(
    model_type: str = "linear",
    epochs: int = 50,
    lr: float = 1e-3,
    device: str = "cuda",
    save_path: str = "checkpoints/classifier_head.pt"
) -> Dict[str, Any]:
    """
    Train a classifier head on real extracted features.
    
    Args:
        model_type: "linear", "mlp", or "attention"
        epochs: Training epochs
        lr: Learning rate
        device: Device for feature extraction
        save_path: Model save path
        
    Returns:
        Training results
    """
    # Prepare data from real dataset
    train_loader, val_loader, all_features = prepare_data(device=device)
    
    num_features = all_features[0].num_features
    
    # Create model
    if model_type == "linear":
        model = LinearClassifierHead(num_features=num_features)
    elif model_type == "mlp":
        model = MLPClassifierHead(num_features=num_features)
    elif model_type == "attention":
        model = AttentionClassifierHead(num_features=num_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"\nModel Architecture: {model_type.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = ClassifierTrainer(model, lr=lr)
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        save_path=save_path
    )
    
    # Print learned weights for linear model
    if model_type == "linear" and isinstance(model, LinearClassifierHead):
        print("\n" + "="*50)
        print("LEARNED SIGNAL WEIGHTS")
        print("="*50)
        weights = model.get_weights()
        for name, weight in weights.items():
            direction = "→ MORE CONTRADICTION" if weight < 0 else "→ MORE CONSISTENT"
            print(f"  {name:30s}: {weight:+.4f} {direction}")
    
    return results


def export_weights(
    checkpoint_path: str = "checkpoints/classifier_head.pt",
    output_path: str = "checkpoints/learned_weights.json"
) -> Dict[str, float]:
    """
    Export learned weights for use in the main pipeline.
    
    Returns:
        Dictionary of weights
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = LinearClassifierHead()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    weights = model.get_weights()
    
    # Save to JSON
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    print(f"Weights exported to {output_path}")
    print("\nLearned Weights:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    return weights


def run_ablation_study(epochs: int = 30, device: str = "cuda") -> Dict[str, Any]:
    """
    Run ablation study to measure each signal's contribution.
    
    Tests classifier with each signal zeroed out.
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Signal Contribution Analysis")
    print("="*60)
    
    results = {}
    
    # Full model
    print("\n[1/6] Full Model (all signals)")
    full_results = train_classifier(
        model_type="linear",
        epochs=epochs,
        device=device,
        save_path="checkpoints/ablation_full.pt"
    )
    results['full'] = full_results['best_accuracy']
    
    signal_names = [
        'nli_score', 'bdh_drift', 'evidence_score',
        'temporal_violations', 'causal_violations'
    ]
    
    # Note: Full ablation requires retraining with masked features
    print("\nNote: Full ablation study requires multiple training runs")
    print("with individual signals masked. Use --mode train with")
    print("custom feature masking for complete analysis.")
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<30} {'Accuracy':>10} {'Δ from Full':>12}")
    print("-"*60)
    
    full_acc = results['full']
    for config, acc in sorted(results.items(), key=lambda x: -x[1]):
        delta = acc - full_acc
        print(f"{config:<30} {acc:>10.4f} {delta:>+12.4f}")
    
    return results


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train/evaluate classifier head for multi-signal fusion"
    )
    parser.add_argument(
        "--mode", 
        type=str, 
        default="train",
        choices=["train", "eval", "export", "ablation"],
        help="Mode: train, eval, export weights, or ablation study"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="linear",
        choices=["linear", "mlp", "attention"],
        help="Classifier architecture"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=50,
        help="Training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda/cpu)"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="checkpoints/classifier_head.pt",
        help="Checkpoint path"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_classifier(
            model_type=args.model,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
            save_path=args.checkpoint
        )
        
    elif args.mode == "eval":
        # Load and evaluate
        _, val_loader, _ = prepare_data(device=args.device)
        
        if args.model == "linear":
            model = LinearClassifierHead()
        elif args.model == "mlp":
            model = MLPClassifierHead()
        else:
            model = AttentionClassifierHead()
        
        trainer = ClassifierTrainer(model)
        trainer.load_model(args.checkpoint)
        
        metrics = trainer.evaluate(val_loader)
        print("\nEvaluation Results:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")
            
    elif args.mode == "export":
        export_weights(args.checkpoint)
        
    elif args.mode == "ablation":
        run_ablation_study(epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
