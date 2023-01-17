from .core import (
    FeaturesData, EFstrType, EShapCalcType, EFeaturesSelectionAlgorithm, EFeaturesSelectionGrouping,
    Pool, CatBoost, CatBoostClassifier, CatBoostRegressor, CatBoostRanker, CatBoostError, cv, train,
    sum_models, _have_equal_features, to_regressor, to_classifier, to_ranker, MultiRegressionCustomMetric,
    MultiRegressionCustomObjective, MultiTargetCustomMetric, MultiTargetCustomObjective
)  # noqa
from .version import VERSION as __version__  # noqa
__all__ = [
    'FeaturesData', 'EFstrType', 'EShapCalcType', 'EFeaturesSelectionAlgorithm', 'EFeaturesSelectionGrouping',
    'Pool', 'CatBoost', 'CatBoostClassifier', 'CatBoostRegressor', 'CatBoostRanker', 'CatboostError',
    'cv', 'train', 'sum_models', '_have_equal_features',
    'to_regressor', 'to_classifier', 'to_ranker', 'MultiRegressionCustomMetric', 'MultiRegressionCustomObjective',
    'MultiTargetCustomMetric', 'MultiTargetCustomObjective'
]

# API compatibility alias.
CatboostError = CatBoostError

try:
    from .widget import MetricVisualizer, MetricsPlotter, XGBPlottingCallback, lgbm_plotting_callback  # noqa
    __all__.extend(['MetricVisualizer', 'MetricsPlotter', 'XGBPlottingCallback', 'lgbm_plotting_callback'])
except:
    pass
