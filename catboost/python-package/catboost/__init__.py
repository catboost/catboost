from .core import FeaturesData, EFstrType, Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatboostError, cv, train, sum_models  # noqa
from .version import VERSION as __version__  # noqa
__all__ = ['FeaturesData', 'EFstrType', 'Pool', 'CatBoost', 'CatBoostClassifier', 'CatBoostRegressor', 'CatboostError', 'cv', 'train', 'sum_models']

try:
    from .widget import MetricVisualizer  # noqa
    __all__.append('MetricVisualizer')
except:
    pass
