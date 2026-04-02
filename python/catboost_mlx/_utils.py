"""
_utils.py -- Shared utility functions for the catboost_mlx package.

What this file does:
    Contains small helper functions used by multiple modules in the package.
    Centralizes them here to avoid duplication.

How it fits into the project:
    Imported by core.py and pool.py. Has no dependencies on other
    catboost_mlx modules.
"""

import numpy as np


def _to_numpy(data) -> np.ndarray:
    """Convert input data to numpy array, handling pandas DataFrames/Series."""
    if isinstance(data, np.ndarray):
        return data
    try:
        import pandas as pd
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
    except ImportError:
        pass
    return np.asarray(data)
