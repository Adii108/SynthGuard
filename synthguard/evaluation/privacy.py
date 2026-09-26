"""
Privacy Risk Analysis for Synthetic Tabular Data.
Evaluates Exact Match / Duplicate Rate, Distance to Closest Record (DCR),
Nearest Neighbor Distance Ratio (NNDR), and Empirical Memorization Risks.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

@dataclass
class PrivacyRiskResult:
    exact_matches_count: int
    exact_match_percentage: float
    min_dcr: float
    mean_dcr: float
    dcr_5th_percentile: float
    mean_nndr: float
    potential_memorization_count: int
    risk_level: str  # 'Low Observed Risk', 'Moderate Caution Required', 'Elevated Privacy Risk'
    key_findings: List[str]
    recommendations: List[str]
    dcr_distribution_sample: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PrivacyRiskEvaluator:
    """
    Computes empirical privacy and memorization risk metrics on synthetic vs real datasets.
    """

    def __init__(self, dcr_memorization_threshold: float = 0.05, nndr_threshold: float = 0.20):
        self.dcr_memorization_threshold = dcr_memorization_threshold
        self.nndr_threshold = nndr_threshold

    def evaluate(self, df_real: pd.DataFrame, df_synth: pd.DataFrame) -> PrivacyRiskResult:
        common_cols = [c for c in df_real.columns if c in df_synth.columns]
        if not common_cols:
            raise ValueError("No common columns between real and synthetic datasets.")

        n_real = len(df_real)
        n_synth = len(df_synth)

        # 1. Exact Row Matches Check
        # Convert rows to string tuples for fast hash set lookup
        real_tuples = set(tuple(x) for x in df_real[common_cols].astype(str).values)
        synth_tuples = [tuple(x) for x in df_synth[common_cols].astype(str).values]

        exact_matches = sum(1 for row in synth_tuples if row in real_tuples)
        exact_pct = round((exact_matches / max(1, n_synth)) * 100, 2)

        # 2. Distance to Closest Record (DCR) in Normalized Feature Space
        # Encode features into numerical space [0, 1]
        real_num = pd.DataFrame()
        synth_num = pd.DataFrame()

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(df_real[col]):
                r_vals = pd.to_numeric(df_real[col], errors='coerce')
                s_vals = pd.to_numeric(df_synth[col], errors='coerce')
                real_num[col] = r_vals.fillna(r_vals.median() if not np.isnan(r_vals.median()) else 0)
                synth_num[col] = s_vals.fillna(s_vals.median() if not np.isnan(s_vals.median()) else 0)
            else:
                cats = sorted(list(set(df_real[col].dropna().astype(str).unique())))
                cat_map = {c: i for i, c in enumerate(cats)}
                real_num[col] = df_real[col].astype(str).map(lambda x: cat_map.get(x, 0))
                synth_num[col] = df_synth[col].astype(str).map(lambda x: cat_map.get(x, 0))

        scaler = MinMaxScaler()
        X_real_norm = scaler.fit_transform(real_num.fillna(0))
        X_synth_norm = scaler.transform(synth_num.fillna(0))

        # Fit NearestNeighbors on Real data (k=2 neighbors to compute DCR & NNDR)
        k_neighbors = min(2, len(X_real_norm))
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean')
        nn.fit(X_real_norm)

        distances, indices = nn.kneighbors(X_synth_norm)

        # d_1 is distance to 1st nearest real neighbor (DCR)
        dcr_values = distances[:, 0]
        min_dcr = float(np.min(dcr_values))
        mean_dcr = float(np.mean(dcr_values))
        p5_dcr = float(np.percentile(dcr_values, 5))

        # NNDR = d_1 / d_2 (ratio to 2nd nearest neighbor; values close to 0 imply isolated memorized points)
        if k_neighbors >= 2:
            d2_values = np.maximum(distances[:, 1], 1e-6)
            nndr_values = dcr_values / d2_values
            mean_nndr = float(np.mean(nndr_values))
            potential_memorized = int(np.sum((dcr_values < self.dcr_memorization_threshold) & (nndr_values < self.nndr_threshold)))
        else:
            mean_nndr = 1.0
            potential_memorized = int(np.sum(dcr_values < self.dcr_memorization_threshold))

        findings = []
        recommendations = []

        if exact_matches > 0:
            findings.append(f"Detected {exact_matches} exact duplicate row(s) ({exact_pct}% of synthetic dataset) matching real records.")
            recommendations.append("Consider applying differential privacy jitter or deduplication filtering before deployment.")
        else:
            findings.append("Zero (0) exact duplicate records found between synthetic and training datasets.")

        if p5_dcr < self.dcr_memorization_threshold:
            findings.append(f"5th percentile Distance-to-Closest-Record is low ({p5_dcr:.4f}), indicating a small cluster of near-duplicate records.")
            recommendations.append("Audit records in the bottom 5% DCR tail for sensitive attribute disclosure.")
        else:
            findings.append(f"Healthy geometric separation observed: 5th percentile DCR is {p5_dcr:.4f}, mean DCR is {mean_dcr:.4f}.")

        if potential_memorized > 0:
            findings.append(f"Identified {potential_memorized} record(s) exhibiting potential memorization signatures (low DCR & low NNDR).")
            recommendations.append("Apply post-generation outlier suppression or noise injection to boundary points.")
        else:
            findings.append("No isolated memorization anomalies detected based on NNDR ratios.")

        # Nuanced Risk Level
        if exact_pct > 5.0 or potential_memorized > (0.05 * n_synth):
            risk_level = "Elevated Privacy Risk"
        elif exact_pct > 0.5 or p5_dcr < self.dcr_memorization_threshold or potential_memorized > 0:
            risk_level = "Moderate Caution Required"
        else:
            risk_level = "Low Observed Risk"

        recommendations.append("Note: Empirical absence of exact matches does not guarantee mathematical differential privacy. Perform secondary domain checks for re-identification risks.")

        # Sample DCR for histogram plotting
        dcr_sample = [round(float(x), 4) for x in np.random.choice(dcr_values, size=min(100, len(dcr_values)), replace=False)]

        return PrivacyRiskResult(
            exact_matches_count=exact_matches,
            exact_match_percentage=exact_pct,
            min_dcr=round(min_dcr, 4),
            mean_dcr=round(mean_dcr, 4),
            dcr_5th_percentile=round(p5_dcr, 4),
            mean_nndr=round(mean_nndr, 4),
            potential_memorization_count=potential_memorized,
            risk_level=risk_level,
            key_findings=findings,
            recommendations=recommendations,
            dcr_distribution_sample=dcr_sample
        )
