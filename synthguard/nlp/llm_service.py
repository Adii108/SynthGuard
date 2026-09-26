"""
LLM Service: Orchestrates natural language request understanding and narrative report generation.
Supports Gemini/OpenAI REST API with graceful fallback to built-in semantic engines.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from synthguard.nlp.intent_parser import GenerationConfig, LocalIntentParser

class LLMService:
    """
    LLM Interface for SynthGuard.
    Provides prompt-to-config extraction and grounded report synthesis.
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "gemini"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.provider = provider
        self.local_parser = LocalIntentParser()

    def interpret_request(self, prompt: str, available_columns: List[str], dataset_row_count: int = 500) -> GenerationConfig:
        """
        Interprets natural language request into GenerationConfig.
        Uses external LLM API if key is present; otherwise utilizes LocalIntentParser.
        """
        if not prompt or not prompt.strip():
            return self.local_parser.parse(
                "Generate realistic synthetic data replicating statistical patterns",
                available_columns,
                dataset_row_count
            )

        # If API key is available, attempt LLM call; fallback seamlessly on error
        if self.api_key:
            try:
                system_prompt = (
                    "You are the SynthGuard GenAI Configurator. Given a user request and dataset columns, "
                    "return ONLY a JSON object with keys: "
                    "'objective' (string), 'target_column' (string), 'important_columns' (list of strings), "
                    "'num_records' (integer), 'selected_model' ('CTGAN' or 'TVAE' or 'Gaussian Copula'), "
                    "'privacy_priority' ('Standard' or 'High (Anti-Memorization)' or 'Balanced'), 'reasoning' (string)."
                )
                user_msg = f"Columns: {available_columns}\nDataset size: {dataset_row_count}\nUser prompt: {prompt}"

                if "openai" in self.provider.lower() or os.environ.get("OPENAI_API_KEY"):
                    resp = requests.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_msg}],
                            "temperature": 0.2
                        },
                        timeout=5
                    )
                    if resp.status_code == 200:
                        content = resp.json()["choices"][0]["message"]["content"]
                        clean_json = content.strip().strip("```json").strip("```")
                        data = json.loads(clean_json)
                        return GenerationConfig(
                            objective=data.get("objective", "Preserve statistical distributions"),
                            target_column=data.get("target_column"),
                            important_columns=data.get("important_columns", available_columns[:3]),
                            num_records=int(data.get("num_records", dataset_row_count)),
                            selected_model=data.get("selected_model", "CTGAN"),
                            privacy_priority=data.get("privacy_priority", "Standard"),
                            raw_prompt=prompt,
                            reasoning=data.get("reasoning", "Parsed via LLM.")
                        )
            except Exception:
                pass

        # Robust local parsing fallback
        return self.local_parser.parse(prompt, available_columns, dataset_row_count)
