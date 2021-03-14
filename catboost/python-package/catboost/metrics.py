# module datasets

from . import _catboost

_dummy_metrics = _catboost.DummyMetrics

class BuiltinMetric:
    pass
