"""
Evaluation module for SynthGuard.
Provides statistical similarity, correlation preservation, ML utility (TSTR), and privacy risk evaluation.
"""
from synthguard.evaluation.statistical import StatisticalEvaluator, StatisticalEvaluationResult, ColumnStatisticalScore
from synthguard.evaluation.correlation import CorrelationEvaluator, CorrelationEvaluationResult
from synthguard.evaluation.ml_utility import MLUtilityEvaluator, MLUtilityResult
from synthguard.evaluation.privacy import PrivacyRiskEvaluator, PrivacyRiskResult

__all__ = [
    "StatisticalEvaluator",
    "StatisticalEvaluationResult",
    "ColumnStatisticalScore",
    "CorrelationEvaluator",
    "CorrelationEvaluationResult",
    "MLUtilityEvaluator",
    "MLUtilityResult",
    "PrivacyRiskEvaluator",
    "PrivacyRiskResult"
]
