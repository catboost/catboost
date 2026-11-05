from ._catboost import CatBoostError

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    _CatBoostNotFittedError = CatBoostError
else:
    class _CatBoostNotFittedError(CatBoostError, NotFittedError):
        """Keep unfitted prediction errors compatible with both libraries."""
