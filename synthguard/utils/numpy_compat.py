"""
NumPy 2.0+ backward compatibility layer for legacy downstream libraries (Plotly, xarray, etc.).
"""
import numpy as np

def apply_numpy_compat():
    """Restores deprecated/removed NumPy 1.x aliases for backward compatibility."""
    aliases = {
        "bool8": getattr(np, "bool_", bool),
        "int0": getattr(np, "intp", int),
        "uint0": getattr(np, "uintp", int),
        "unicode_": getattr(np, "str_", str),
        "string_": getattr(np, "bytes_", bytes),
        "bytes0": getattr(np, "bytes_", bytes),
        "str0": getattr(np, "str_", str),
        "object0": getattr(np, "object_", object),
        "float_": getattr(np, "float64", float),
        "complex_": getattr(np, "complex128", complex),
    }
    for alias, target in aliases.items():
        if not hasattr(np, alias):
            try:
                setattr(np, alias, target)
            except Exception:
                pass

apply_numpy_compat()
