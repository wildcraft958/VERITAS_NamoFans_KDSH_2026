"""
Training Package for KDSH 2026 BDH

Modules:
- classifier_head: Train lightweight classifier on multi-signal features
"""

from .classifier_head import (
    LinearClassifierHead,
    MLPClassifierHead,
    AttentionClassifierHead,
    SignalFeatures,
    SignalDataset,
    ClassifierTrainer,
    FeatureExtractor,
    train_classifier,
    export_weights,
)

__all__ = [
    'LinearClassifierHead',
    'MLPClassifierHead',
    'AttentionClassifierHead',
    'SignalFeatures',
    'SignalDataset',
    'ClassifierTrainer',
    'FeatureExtractor',
    'train_classifier',
    'export_weights',
]
