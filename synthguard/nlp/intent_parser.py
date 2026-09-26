"""
Natural Language Intent Parser for Synthetic Data Generation Requests.
Translates user natural language instructions into structured generation configurations.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import re
import difflib

@dataclass
class GenerationConfig:
    objective: str
    target_column: Optional[str]
    important_columns: List[str]
    num_records: int
    selected_model: str
    privacy_priority: str  # 'Standard', 'High (Anti-Memorization)', 'Balanced'
    raw_prompt: str
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LocalIntentParser:
    """
    Robust rule-guided & semantic NLP parser to extract structured generation configs
    from natural language requests without requiring an external API key.
    """

    def parse(self, prompt: str, available_columns: List[str], dataset_row_count: int = 500) -> GenerationConfig:
        prompt_lower = prompt.lower()

        # 1. Extract Number of Records
        num_records = dataset_row_count
        # Pattern: e.g. "1000 records", "500 rows", "generate 2000 samples", "size 750"
        num_match = re.search(r'(?:generate|create|sample|make)?\s*(\d{2,6})\s*(?:records|rows|samples|entries|items)?', prompt_lower)
        if num_match:
            try:
                extracted = int(num_match.group(1))
                if 10 <= extracted <= 100000:
                    num_records = extracted
            except Exception:
                pass

        if "double" in prompt_lower or "2x" in prompt_lower:
            num_records = dataset_row_count * 2
        elif "half" in prompt_lower or "0.5x" in prompt_lower:
            num_records = max(50, dataset_row_count // 2)

        # 2. Extract Preferred Model
        selected_model = "CTGAN"
        if "tvae" in prompt_lower or "autoencoder" in prompt_lower or "variational" in prompt_lower:
            selected_model = "TVAE"
        elif "copula" in prompt_lower or "gaussian" in prompt_lower:
            selected_model = "Gaussian Copula"
        elif "ctgan" in prompt_lower or "gan" in prompt_lower:
            selected_model = "CTGAN"

        # 3. Match Important Columns from prompt
        matched_columns = []
        col_map = {c.lower(): c for c in available_columns}

        # Direct token match
        for col_lower, actual_col in col_map.items():
            # Check whole word match or snake_case/space variation
            col_var = col_lower.replace("_", " ")
            if re.search(r'\b' + re.escape(col_lower) + r'\b', prompt_lower) or (col_var != col_lower and col_var in prompt_lower):
                matched_columns.append(actual_col)

        # Fuzzy match for remaining terms if few matched
        if len(matched_columns) < 2 and len(available_columns) > 0:
            words = re.findall(r'[a-zA-Z]{3,}', prompt)
            for w in words:
                close = difflib.get_close_matches(w.lower(), col_map.keys(), n=1, cutoff=0.82)
                if close:
                    actual = col_map[close[0]]
                    if actual not in matched_columns:
                        matched_columns.append(actual)

        if not matched_columns:
            matched_columns = available_columns[:min(4, len(available_columns))]

        # 4. Target Column Detection
        target_column = None
        target_keywords = ["predict", "target", "classify", "attrition", "churn", "outcome", "label", "dependent"]
        for col in available_columns:
            if col.lower() in prompt_lower:
                for kw in target_keywords:
                    if kw in prompt_lower and col.lower() in prompt_lower:
                        target_column = col
                        break

        if not target_column:
            # Check if any available column matches common targets
            for col in available_columns:
                if col.lower() in ["attrition", "churn", "target", "label", "status"]:
                    target_column = col
                    break
            if not target_column and available_columns:
                target_column = available_columns[-1]

        # 5. Privacy Priority
        if any(w in prompt_lower for w in ["privacy", "anonymize", "confidential", "no duplicate", "differential", "sensitive"]):
            privacy_priority = "High (Anti-Memorization)"
        elif any(w in prompt_lower for w in ["utility", "accuracy", "ml", "performance", "high fidelity"]):
            privacy_priority = "Balanced (Fidelity Focus)"
        else:
            privacy_priority = "Standard"

        # 6. Objective Summary
        if "preserve" in prompt_lower and "relationship" in prompt_lower:
            objective = "Preserve multi-variable correlations and joint distributions"
        elif "ml" in prompt_lower or "train" in prompt_lower or "model" in prompt_lower:
            objective = "Optimize synthetic data for downstream machine learning model utility"
        elif "privacy" in prompt_lower:
            objective = "Generate privacy-preserving synthetic records with low memorization risk"
        else:
            objective = "Generate realistic tabular data replicating original statistical patterns"

        reasoning = (
            f"Interpreted request for {num_records} synthetic rows using {selected_model}. "
            f"Preserving key features: {', '.join(matched_columns)}. "
            f"Downstream target designated as '{target_column}' with '{privacy_priority}' privacy posture."
        )

        return GenerationConfig(
            objective=objective,
            target_column=target_column,
            important_columns=matched_columns,
            num_records=num_records,
            selected_model=selected_model,
            privacy_priority=privacy_priority,
            raw_prompt=prompt,
            reasoning=reasoning
        )
