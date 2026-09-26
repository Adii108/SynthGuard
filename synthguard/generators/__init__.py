"""
Generators module for SynthGuard.
"""
from typing import Dict, Type
from synthguard.generators.base import BaseTabularGenerator
from synthguard.generators.copula_generator import GaussianCopulaGenerator
from synthguard.generators.ctgan_generator import CTGANGenerator
from synthguard.generators.tvae_generator import TVAEGenerator

GENERATORS: Dict[str, Type[BaseTabularGenerator]] = {
    "CTGAN": CTGANGenerator,
    "TVAE": TVAEGenerator,
    "Gaussian Copula": GaussianCopulaGenerator,
}

def get_generator(model_name: str, random_state: int = 42, **kwargs) -> BaseTabularGenerator:
    """Factory function to instantiate synthetic data generator."""
    name_clean = model_name.strip()
    # Case-insensitive lookup
    for key, cls in GENERATORS.items():
        if key.lower() == name_clean.lower() or key.lower().replace(" ", "") == name_clean.lower().replace(" ", ""):
            return cls(random_state=random_state, **kwargs)
    raise ValueError(f"Unknown generator model '{model_name}'. Available: {list(GENERATORS.keys())}")

__all__ = [
    "BaseTabularGenerator",
    "GaussianCopulaGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "GENERATORS",
    "get_generator",
]
