"""CatBoost training and inference through Apple Metal."""

from .regressor import CatBoostMetalRegressor, CatBoostMetalClassifier
from .ranker import CatBoostMetalRanker
from ._native import device_info

__all__ = ["CatBoostMetalRegressor", "CatBoostMetalClassifier", "CatBoostMetalRanker", "device_info"]
