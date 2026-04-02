"""CatBoost-MLX: Gradient boosted decision trees on Apple Silicon via Metal/MLX."""

from .core import CatBoostMLX, CatBoostMLXRegressor, CatBoostMLXClassifier

__version__ = "0.1.0"
__all__ = ["CatBoostMLX", "CatBoostMLXRegressor", "CatBoostMLXClassifier"]
