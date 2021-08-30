from .core import (
    FeaturesData, EFstrType, EShapCalcType, EFeaturesSelectionAlgorithm, Pool, CatBoost,
    CatBoostClassifier, CatBoostRegressor, CatBoostRanker, CatBoostError, cv, train, sum_models, _have_equal_features,
    to_regressor, to_classifier, to_ranker, MultiRegressionCustomMetric, MultiRegressionCustomObjective
)  # noqa
from .version import VERSION as __version__  # noqa
__all__ = [
    'FeaturesData', 'EFstrType', 'EShapCalcType', 'EFeaturesSelectionAlgorithm', 'Pool', 'CatBoost',
    'CatBoostClassifier', 'CatBoostRegressor', 'CatBoostRanker', 'CatboostError',
    'cv', 'train', 'sum_models', '_have_equal_features',
    'to_regressor', 'to_classifier', 'to_ranker', 'MultiRegressionCustomMetric', 'MultiRegressionCustomObjective'
]

# API compatibility alias.
CatboostError = CatBoostError

try:
    from .widget import MetricVisualizer  # noqa
    __all__.append('MetricVisualizer')
except:
    pass
