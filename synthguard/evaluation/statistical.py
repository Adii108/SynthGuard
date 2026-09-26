"""
Statistical similarity evaluation for synthetic tabular data.
Computes KS-Test, Wasserstein Distance, Total Variation Distance (TVD), and distribution overlaps.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from scipy import stats

@dataclass
class ColumnStatisticalScore:
    column_name: str
    column_type: str  # 'numerical' or 'categorical'
    similarity_score: float  # [0.0, 1.0] where 1.0 is identical
    metric_name: str
    metric_value: float
    p_value: Optional[float]
    summary_text: str
    real_stats: Dict[str, Any]
    synth_stats: Dict[str, Any]


@dataclass
class StatisticalEvaluationResult:
    overall_similarity_score: float  # [0.0, 1.0]
    column_scores: Dict[str, ColumnStatisticalScore]
    numerical_metrics: Dict[str, Any]
    categorical_metrics: Dict[str, Any]
    summary_insights: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StatisticalEvaluator:
    """
    Evaluates marginal distribution similarity between real and synthetic tabular datasets.
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level

    def evaluate(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> StatisticalEvaluationResult:
        common_cols = [c for c in df_real.columns if c in df_synth.columns]
        if not common_cols:
            raise ValueError("No common columns found between real and synthetic datasets.")

        col_scores: Dict[str, ColumnStatisticalScore] = {}
        numerical_metrics = {}
        categorical_metrics = {}
        similarity_list = []
        insights = []

        for col in common_cols:
            s_real = df_real[col].dropna()
            s_synth = df_synth[col].dropna()

            if s_real.empty or s_synth.empty:
                continue

            is_numeric = pd.api.types.is_numeric_dtype(s_real) and s_real.nunique() > 5

            if is_numeric:
                # Kolmogorov-Smirnov Test
                v_real = s_real.values.astype(float)
                v_synth = s_synth.values.astype(float)

                ks_stat, ks_pval = stats.ks_2samp(v_real, v_synth)
                # 1-Wasserstein Distance (Earth Mover's Distance)
                std_pool = np.std(v_real) + 1e-6
                w_dist = stats.wasserstein_distance(v_real, v_synth)
                norm_w_dist = w_dist / std_pool

                # Continuous similarity score based on 1 - KS statistic
                sim_score = max(0.0, min(1.0, 1.0 - ks_stat))
                similarity_list.append(sim_score)

                # Qualitative explanation
                if sim_score >= 0.90:
                    explanation = f"Distribution is exceptionally well-aligned with real data (KS distance: {ks_stat:.3f})."
                elif sim_score >= 0.75:
                    explanation = f"Distribution shows strong alignment with minor density shifts (KS distance: {ks_stat:.3f})."
                elif sim_score >= 0.55:
                    explanation = f"Moderate distribution drift observed; tail behavior or variance differs (KS distance: {ks_stat:.3f})."
                else:
                    explanation = f"Significant distribution divergence from original dataset (KS distance: {ks_stat:.3f})."

                real_stat = {
                    "mean": round(float(np.mean(v_real)), 2),
                    "std": round(float(np.std(v_real)), 2),
                    "median": round(float(np.median(v_real)), 2),
                }
                synth_stat = {
                    "mean": round(float(np.mean(v_synth)), 2),
                    "std": round(float(np.std(v_synth)), 2),
                    "median": round(float(np.median(v_synth)), 2),
                }

                col_scores[col] = ColumnStatisticalScore(
                    column_name=col,
                    column_type="numerical",
                    similarity_score=round(sim_score, 3),
                    metric_name="Kolmogorov-Smirnov Statistic",
                    metric_value=round(ks_stat, 4),
                    p_value=round(ks_pval, 4) if ks_pval is not None else None,
                    summary_text=explanation,
                    real_stats=real_stat,
                    synth_stats=synth_stat
                )
                numerical_metrics[col] = {
                    "ks_statistic": round(ks_stat, 4),
                    "ks_pvalue": round(ks_pval, 4),
                    "wasserstein_distance": round(w_dist, 3),
                    "normalized_wasserstein": round(norm_w_dist, 3),
                    "similarity": round(sim_score, 3)
                }

            else:
                # Categorical Total Variation Distance (TVD)
                vc_real = s_real.astype(str).value_counts(normalize=True)
                vc_synth = s_synth.astype(str).value_counts(normalize=True)

                all_cats = list(set(vc_real.index).union(set(vc_synth.index)))
                p_real = np.array([vc_real.get(c, 0.0) for c in all_cats])
                p_synth = np.array([vc_synth.get(c, 0.0) for c in all_cats])

                # TVD = 0.5 * sum(|p - q|) in [0, 1]
                tvd = 0.5 * np.sum(np.abs(p_real - p_synth))
                sim_score = max(0.0, min(1.0, 1.0 - tvd))
                similarity_list.append(sim_score)

                if sim_score >= 0.90:
                    explanation = f"Category frequencies closely replicate original proportions (TVD: {tvd:.3f})."
                elif sim_score >= 0.75:
                    explanation = f"Minor frequency discrepancy in secondary categories (TVD: {tvd:.3f})."
                else:
                    explanation = f"Notable category imbalance compared to real dataset (TVD: {tvd:.3f})."

                real_stat = {c: round(float(p), 3) for c, p in zip(all_cats, p_real)}
                synth_stat = {c: round(float(p), 3) for c, p in zip(all_cats, p_synth)}

                col_scores[col] = ColumnStatisticalScore(
                    column_name=col,
                    column_type="categorical",
                    similarity_score=round(sim_score, 3),
                    metric_name="Total Variation Distance (TVD)",
                    metric_value=round(tvd, 4),
                    p_value=None,
                    summary_text=explanation,
                    real_stats=real_stat,
                    synth_stats=synth_stat
                )
                categorical_metrics[col] = {
                    "tvd": round(tvd, 4),
                    "similarity": round(sim_score, 3)
                }

        avg_sim = round(float(np.mean(similarity_list)) if similarity_list else 0.0, 3)

        # Generate automated high-level insights
        top_aligned = sorted(col_scores.items(), key=lambda x: x[1].similarity_score, reverse=True)[:3]
        weak_aligned = sorted(col_scores.items(), key=lambda x: x[1].similarity_score)[:2]

        top_names = ", ".join([f"'{k}' ({v.similarity_score:.0%})" for k, v in top_aligned])
        insights.append(f"Highest statistical fidelity achieved in: {top_names}.")

        if weak_aligned and weak_aligned[0][1].similarity_score < 0.75:
            weak_names = ", ".join([f"'{k}' ({v.similarity_score:.0%})" for k, v in weak_aligned])
            insights.append(f"Moderate variance observed in: {weak_names}.")

        return StatisticalEvaluationResult(
            overall_similarity_score=avg_sim,
            column_scores=col_scores,
            numerical_metrics=numerical_metrics,
            categorical_metrics=categorical_metrics,
            summary_insights=insights
        )
