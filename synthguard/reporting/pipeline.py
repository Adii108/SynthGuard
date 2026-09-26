"""
Automated Evaluation Pipeline.
Coordinates all evaluation engines (Statistical, Correlation, ML Utility, Privacy) into a single cohesive analysis.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
import pandas as pd
from synthguard.evaluation.statistical import StatisticalEvaluator, StatisticalEvaluationResult
from synthguard.evaluation.correlation import CorrelationEvaluator, CorrelationEvaluationResult
from synthguard.evaluation.ml_utility import MLUtilityEvaluator, MLUtilityResult
from synthguard.evaluation.privacy import PrivacyRiskEvaluator, PrivacyRiskResult

@dataclass
class EvaluationSummary:
    statistical: StatisticalEvaluationResult
    correlation: CorrelationEvaluationResult
    ml_utility: MLUtilityResult
    privacy: PrivacyRiskResult
    real_rows: int
    synth_rows: int
    column_count: int
    target_column: str
    selected_model: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvaluationPipeline:
    """
    Automated execution engine that runs full diagnostic benchmarks on synthetic datasets.
    """

    def __init__(self, random_state: int = 42):
        self.statistical_evaluator = StatisticalEvaluator()
        self.correlation_evaluator = CorrelationEvaluator()
        self.ml_utility_evaluator = MLUtilityEvaluator(random_state=random_state)
        self.privacy_evaluator = PrivacyRiskEvaluator()

    def run(self, df_real: pd.DataFrame, df_synth: pd.DataFrame, target_column: Optional[str] = None, selected_model: str = "CTGAN") -> EvaluationSummary:
        """
        Executes end-to-end evaluation suite.
        """
        stat_res = self.statistical_evaluator.evaluate(df_real, df_synth)
        corr_res = self.correlation_evaluator.evaluate(df_real, df_synth)
        ml_res = self.ml_utility_evaluator.evaluate(df_real, df_synth, target_column=target_column)
        priv_res = self.privacy_evaluator.evaluate(df_real, df_synth)

        return EvaluationSummary(
            statistical=stat_res,
            correlation=corr_res,
            ml_utility=ml_res,
            privacy=priv_res,
            real_rows=len(df_real),
            synth_rows=len(df_synth),
            column_count=len(df_real.columns),
            target_column=ml_res.target_column,
            selected_model=selected_model
        )
