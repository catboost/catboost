from .core import FeaturesData, EFstrType, Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostError, cv, train, sum_models, _have_equal_features, to_regressor, to_classifier  # noqa
from .version import VERSION as __version__  # noqa
__all__ = ['FeaturesData', 'EFstrType', 'Pool', 'CatBoost', 'CatBoostClassifier', 'CatBoostRegressor', 'CatBoostError', 'CatboostError', 'cv', 'train', 'sum_models', '_have_equal_features', \
           'to_regressor', 'to_classifier']

# API compatibility alias.
CatboostError = CatBoostError

try:
    from .widget import MetricVisualizer  # noqa
    __all__.append('MetricVisualizer')
except:
    pass
