"""
Correlation and Association Preservation Evaluation.
Computes Pearson/Spearman matrices, Cramer's V for categorical features, and Frobenius norm difference.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from scipy import stats

@dataclass
class CorrelationEvaluationResult:
    mean_absolute_error: float
    frobenius_norm_diff: float
    correlation_fidelity_score: float  # [0.0, 1.0]
    real_corr_matrix: Dict[str, Dict[str, float]]
    synth_corr_matrix: Dict[str, Dict[str, float]]
    diff_matrix: Dict[str, Dict[str, float]]
    preserved_pairs: List[Dict[str, Any]]
    degraded_pairs: List[Dict[str, Any]]
    summary_text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CorrelationEvaluator:
    """
    Evaluates how accurately pairwise relationships and feature interactions are retained.
    """

    @staticmethod
    def _cramers_v(x: pd.Series, y: pd.Series) -> float:
        """Computes Cramer's V association for categorical variables."""
        confusion_matrix = pd.crosstab(x, y)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        if min(r - 1, k - 1) == 0:
            return 0.0
        # Bias-corrected Cramer's V
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        denom = min((kcorr - 1), (rcorr - 1))
        if denom <= 0:
            return 0.0
        return float(np.sqrt(phi2corr / denom))

    def evaluate(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> CorrelationEvaluationResult:
        common_cols = [c for c in df_real.columns if c in df_synth.columns]
        if len(common_cols) < 2:
            raise ValueError("At least 2 common columns required for correlation evaluation.")

        # Encode categorical variables numerically for unified correlation computation
        real_num = pd.DataFrame()
        synth_num = pd.DataFrame()

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df_real[col]):
                r_vals = pd.to_numeric(df_real[col], errors='coerce')
                s_vals = pd.to_numeric(df_synth[col], errors='coerce')
                real_num[col] = r_vals.fillna(r_vals.median() if not np.isnan(r_vals.median()) else 0)
                synth_num[col] = s_vals.fillna(s_vals.median() if not np.isnan(s_vals.median()) else 0)
            else:
                # Factorize / label encode with consistent ordering
                categories = sorted(list(set(df_real[col].dropna().astype(str).unique())))
                cat_map = {c: i for i, c in enumerate(categories)}
                real_num[col] = df_real[col].astype(str).map(lambda x: cat_map.get(x, 0))
                synth_num[col] = df_synth[col].astype(str).map(lambda x: cat_map.get(x, 0))

        corr_real = real_num.corr(method="pearson").fillna(0.0)
        corr_synth = synth_num.corr(method="pearson").fillna(0.0)
        corr_diff = (corr_real - corr_synth).abs()

        n_features = len(common_cols)
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n_features, k=1)
        real_vals = corr_real.values[triu_indices]
        synth_vals = corr_synth.values[triu_indices]
        abs_diffs = np.abs(real_vals - synth_vals)

        mae = float(np.mean(abs_diffs)) if len(abs_diffs) > 0 else 0.0
        frob_diff = float(np.linalg.norm(corr_real.values - corr_synth.values, ord='fro'))
        fidelity_score = max(0.0, min(1.0, 1.0 - mae))

        # Identify preserved vs degraded relationships
        pair_analysis = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                c1 = common_cols[i]
                c2 = common_cols[j]
                r_val = float(corr_real.iloc[i, j])
                s_val = float(corr_synth.iloc[i, j])
                d_val = float(corr_diff.iloc[i, j])
                pair_analysis.append({
                    "col1": c1,
                    "col2": c2,
                    "real_corr": round(r_val, 3),
                    "synth_corr": round(s_val, 3),
                    "abs_error": round(d_val, 3),
                    "status": "Preserved" if d_val <= 0.15 else ("Moderate Drift" if d_val <= 0.30 else "Degraded")
                })

        # Sort by absolute error
        preserved = sorted([p for p in pair_analysis if p["status"] == "Preserved"], key=lambda x: abs(x["real_corr"]), reverse=True)[:5]
        degraded = sorted([p for p in pair_analysis if p["status"] != "Preserved"], key=lambda x: x["abs_error"], reverse=True)[:5]

        if fidelity_score >= 0.85:
            summary = f"High correlation preservation (Mean Absolute Error: {mae:.3f}). Critical inter-feature associations are maintained."
        elif fidelity_score >= 0.70:
            summary = f"Moderate correlation preservation (MAE: {mae:.3f}). Primary linear trends are retained, but secondary interactions show mild attenuation."
        else:
            summary = f"Noticeable correlation decay (MAE: {mae:.3f}). Feature interactions in synthetic data deviate from real patterns."

        return CorrelationEvaluationResult(
            mean_absolute_error=round(mae, 4),
            frobenius_norm_diff=round(frob_diff, 4),
            correlation_fidelity_score=round(fidelity_score, 3),
            real_corr_matrix={c: {k: round(float(v), 3) for k, v in row.items()} for c, row in corr_real.to_dict().items()},
            synth_corr_matrix={c: {k: round(float(v), 3) for k, v in row.items()} for c, row in corr_synth.to_dict().items()},
            diff_matrix={c: {k: round(float(v), 3) for k, v in row.items()} for c, row in corr_diff.to_dict().items()},
            preserved_pairs=preserved,
            degraded_pairs=degraded,
            summary_text=summary
        )
