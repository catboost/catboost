"""
__init__.py -- Package entry point for CatBoost-MLX.

What this file does:
    When Python imports the catboost_mlx package, it runs this file first.
    It gathers the main classes from internal modules and re-exports them,
    so users can write ``from catboost_mlx import CatBoostMLXRegressor``
    without knowing which internal file defines it.

How it fits into the project:
    This is the front door. It imports from core.py (model classes) and
    pool.py (data container). Every user interaction starts here.

Key concepts:
    - Package entry point: Python runs __init__.py when a folder is imported.
    - Re-export: Making internal classes available at the top-level package name.
    - __all__: Controls what ``from catboost_mlx import *`` exposes.
"""

# Model classes (the main API users interact with)
from .core import CatBoostMLX, CatBoostMLXRegressor, CatBoostMLXClassifier

# Data container for bundling features + labels + metadata
from .pool import Pool

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("catboost-mlx")
except Exception:
    __version__ = "0.1.0"  # fallback for editable installs / development

__all__ = ["CatBoostMLX", "CatBoostMLXRegressor", "CatBoostMLXClassifier", "Pool"]
