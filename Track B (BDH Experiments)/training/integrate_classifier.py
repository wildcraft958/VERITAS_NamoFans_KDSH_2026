"""
Integrate Trained Classifier Head with SOTA Pipeline
=====================================================

This script demonstrates how to use the trained classifier head
to replace hand-tuned signal weights in the main pipeline.

Usage:
    # Train classifier first
    python training/classifier_head.py --mode train --epochs 50
    
    # Then use this integration
    python training/integrate_classifier.py --checkpoint checkpoints/classifier_head.pt
"""

import os
import sys
import json
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.classifier_head import (
    LinearClassifierHead,
    MLPClassifierHead,
    AttentionClassifierHead,
    SignalFeatures,
)


class LearnedWeightPipeline:
    """
    Wrapper that uses learned classifier weights instead of hand-tuned.
    
    Integrates with SOTANarrativeConsistencyPipeline but replaces
    the weighted sum with a trained classifier.
    """
    
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/classifier_head.pt",
        model_type: str = "linear",
        device: str = "cuda",
        load_full_pipeline: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        # Load classifier
        self.classifier = self._load_classifier(model_type)
        
        # Optionally load full pipeline
        self.pipeline = None
        if load_full_pipeline:
            self._load_full_pipeline()
    
    def _load_classifier(self, model_type: str) -> nn.Module:
        """Load trained classifier from checkpoint."""
        if model_type == "linear":
            model = LinearClassifierHead()
        elif model_type == "mlp":
            model = MLPClassifierHead()
        elif model_type == "attention":
            model = AttentionClassifierHead()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        if os.path.exists(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded classifier from {self.checkpoint_path}")
            print(f"  Best accuracy during training: {checkpoint.get('best_accuracy', 'N/A'):.4f}")
        else:
            print(f"⚠ Checkpoint not found: {self.checkpoint_path}")
            print("  Using initialized weights (not trained)")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_full_pipeline(self):
        """Load the full SOTA pipeline."""
        print("\nLoading SOTA Pipeline components...")
        from src.pipeline.sota_pipeline import SOTANarrativeConsistencyPipeline
        self.pipeline = SOTANarrativeConsistencyPipeline(device=str(self.device))
    
    def extract_signal_features(
        self,
        novel_text: str,
        backstory: str
    ) -> SignalFeatures:
        """
        Extract raw signal features from pipeline.
        
        Returns SignalFeatures that can be fed to classifier.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Set load_full_pipeline=True")
        
        # Ensure novel is ingested
        if self.pipeline.novel_text != novel_text:
            self.pipeline.ingest_novel(novel_text)
        
        # Get verdict from pipeline (uses hand-tuned weights)
        verdict = self.pipeline.verify_backstory(backstory)
        
        # Extract raw features
        features = SignalFeatures(
            sample_id="",
            book_name="",
            nli_score=verdict.nli_score,
            bdh_drift=verdict.perplexity_score,
            evidence_score=1.0 - verdict.nli_score,  # Inverse
            temporal_violations=verdict.temporal_violations,
            causal_violations=verdict.causal_violations,
            nli_confidence=verdict.confidence,
            contradiction_evidence_count=len(verdict.contradicting_evidence),
            supporting_evidence_count=len(verdict.supporting_evidence),
        )
        
        return features
    
    def predict(
        self,
        novel_text: str,
        backstory: str,
        return_features: bool = False
    ) -> Dict[str, Any]:
        """
        Make prediction using trained classifier.
        
        Args:
            novel_text: Full novel text
            backstory: Character backstory to verify
            return_features: Include raw features in output
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extract_signal_features(novel_text, backstory)
        
        # Convert to tensor
        x = features.to_tensor().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            prob = self.classifier(x).item()
        
        # Decision
        consistent = prob > 0.5
        confidence = prob if consistent else 1.0 - prob
        
        result = {
            'consistent': consistent,
            'label': 1 if consistent else 0,
            'probability': prob,
            'confidence': confidence,
        }
        
        if return_features:
            result['features'] = features.to_dict()
        
        return result
    
    def predict_from_features(
        self,
        features: SignalFeatures
    ) -> Dict[str, Any]:
        """
        Make prediction from pre-extracted features.
        
        Useful for batch processing where features are already extracted.
        """
        x = features.to_tensor().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.classifier(x).item()
        
        consistent = prob > 0.5
        
        return {
            'consistent': consistent,
            'label': 1 if consistent else 0,
            'probability': prob,
            'confidence': prob if consistent else 1.0 - prob,
        }
    
    def get_learned_weights(self) -> Optional[Dict[str, float]]:
        """
        Get learned weights if using linear classifier.
        
        Returns None for non-linear classifiers.
        """
        if isinstance(self.classifier, LinearClassifierHead):
            return self.classifier.get_weights()
        return None
    
    def print_learned_weights(self):
        """Print learned weights in human-readable format."""
        weights = self.get_learned_weights()
        
        if weights is None:
            print("Non-linear classifier - weights not directly interpretable")
            return
        
        print("\n" + "="*60)
        print("LEARNED SIGNAL WEIGHTS")
        print("="*60)
        print(f"{'Signal':<30} {'Weight':>10} {'Interpretation':<25}")
        print("-"*60)
        
        interpretations = {
            'nli_score': ('Higher → Contradiction', weights['nli_score'] < 0),
            'bdh_drift': ('Higher → Contradiction', weights['bdh_drift'] < 0),
            'evidence_score': ('Higher → Consistent', weights['evidence_score'] > 0),
            'temporal_violations': ('More → Contradiction', weights['temporal_violations'] < 0),
            'causal_violations': ('More → Contradiction', weights['causal_violations'] < 0),
            'nli_confidence': ('Higher → Trust NLI', abs(weights['nli_confidence']) > 0.1),
            'contradiction_count': ('More → Contradiction', weights['contradiction_count'] < 0),
            'supporting_count': ('More → Consistent', weights['supporting_count'] > 0),
            'bias': ('Base prior', True),
        }
        
        for name, weight in weights.items():
            interp, expected = interpretations.get(name, ('', True))
            status = "✓" if expected else "⚠"
            print(f"{name:<30} {weight:>+10.4f} {status} {interp}")
        
        print("="*60)


def compare_handtuned_vs_learned(
    novel_text: str,
    backstory: str,
    checkpoint_path: str = "checkpoints/classifier_head.pt"
):
    """
    Compare hand-tuned pipeline vs learned classifier.
    """
    print("\n" + "="*60)
    print("COMPARISON: Hand-Tuned vs Learned Weights")
    print("="*60)
    
    # Load learned pipeline
    learned_pipeline = LearnedWeightPipeline(
        checkpoint_path=checkpoint_path,
        load_full_pipeline=True
    )
    
    # Get hand-tuned result from original pipeline
    handtuned_verdict = learned_pipeline.pipeline.verify_backstory(backstory)
    
    # Get learned result
    learned_result = learned_pipeline.predict(novel_text, backstory, return_features=True)
    
    print("\nHand-Tuned Pipeline:")
    print(f"  Verdict: {'Consistent' if handtuned_verdict.consistent else 'Contradict'}")
    print(f"  Confidence: {handtuned_verdict.confidence:.4f}")
    print(f"  NLI Score: {handtuned_verdict.nli_score:.4f}")
    print(f"  BDH Drift: {handtuned_verdict.perplexity_score:.4f}")
    
    print("\nLearned Classifier:")
    print(f"  Verdict: {'Consistent' if learned_result['consistent'] else 'Contradict'}")
    print(f"  Probability: {learned_result['probability']:.4f}")
    print(f"  Confidence: {learned_result['confidence']:.4f}")
    
    print("\nRaw Features Used:")
    features = learned_result['features']
    for key, value in features.items():
        if key not in ['sample_id', 'book_name', 'label']:
            print(f"  {key}: {value}")
    
    # Agreement
    agree = handtuned_verdict.consistent == learned_result['consistent']
    print(f"\nAgreement: {'✓ SAME' if agree else '✗ DIFFERENT'}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrate trained classifier with SOTA pipeline"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/classifier_head.pt",
        help="Path to trained classifier checkpoint"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="linear",
        choices=["linear", "mlp", "attention"],
        help="Classifier architecture"
    )
    parser.add_argument(
        "--show-weights",
        action="store_true",
        help="Display learned weights"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo comparison"
    )
    
    args = parser.parse_args()
    
    if args.show_weights:
        pipeline = LearnedWeightPipeline(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
            load_full_pipeline=False
        )
        pipeline.print_learned_weights()
    
    elif args.demo:
        print("Demo mode requires a novel and backstory - showing weight analysis instead")
        pipeline = LearnedWeightPipeline(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
            load_full_pipeline=False
        )
        pipeline.print_learned_weights()
    
    else:
        # Just show weights by default
        pipeline = LearnedWeightPipeline(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
            load_full_pipeline=False
        )
        pipeline.print_learned_weights()
        
        print("\n" + "-"*60)
        print("Usage Examples:")
        print("-"*60)
        print("""
# In your code:
from training.integrate_classifier import LearnedWeightPipeline

pipeline = LearnedWeightPipeline(
    checkpoint_path='checkpoints/classifier_head.pt',
    load_full_pipeline=True
)

result = pipeline.predict(novel_text, backstory)
print(f"Consistent: {result['consistent']}, Confidence: {result['confidence']:.4f}")
""")


if __name__ == "__main__":
    main()
