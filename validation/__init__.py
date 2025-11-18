"""
Validation Package
==================

Advanced validation framework for trading strategies.

Modules:
--------
- confidence_filter: Adaptive probability-based filtering
- optimize_confidence_threshold: Automatic threshold optimization
- purged_kfold: Cross-validation without information leakage
- ensemble_scoring: Multi-factor trade scoring system
- run_advanced_validation: Complete validation pipeline

Usage:
------
    from validation.confidence_filter import ConfidenceFilter
    from validation.optimize_confidence_threshold import ThresholdOptimizer
    from validation.purged_kfold import PurgedKFold
    from validation.ensemble_scoring import EnsembleScorer
"""

from .confidence_filter import ConfidenceFilter, DynamicConfidenceFilter
from .optimize_confidence_threshold import ThresholdOptimizer
from .purged_kfold import PurgedKFold, CombinatorialPurgedCV, PurgedWalkForward
from .ensemble_scoring import EnsembleScorer, TradeSignal, TradeQuality

__version__ = '1.0.0'
__all__ = [
    'ConfidenceFilter',
    'DynamicConfidenceFilter',
    'ThresholdOptimizer',
    'PurgedKFold',
    'CombinatorialPurgedCV',
    'PurgedWalkForward',
    'EnsembleScorer',
    'TradeSignal',
    'TradeQuality',
]
