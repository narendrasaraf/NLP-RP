"""
backend/utils.py
----------------
Helper functions for the cognitive backend.
"""

def normalize_value(val: float, min_val: float, max_val: float) -> float:
    """
    Normalizes a value to be strictly between 0.0 and 1.0.
    """
    if max_val == min_val:
        return 0.0
    
    normalized = (val - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))

def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamps a value within a specified range."""
    return max(min_val, min(val, max_val))
