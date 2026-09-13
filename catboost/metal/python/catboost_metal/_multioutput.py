"""Vector targets on the shared Metal tree engine, with no class gauge.

Targets are row-major[N,D], except uncertainty uses a scalar target[N].
All dimensions share each selected structure; this is not separate ensembles.
"""
import numpy as np
from . import _multiclass


def _dimensions(targets, objective, dimensions, classes):
    if dimensions is not None and classes is not None and dimensions != classes:
        raise ValueError("dimensions and classes aliases disagree.")
    result = dimensions if dimensions is not None else classes
    if objective not in ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy"):
        raise ValueError("Unsupported multioutput objective.")
    if result is None:
        values = np.asarray(targets)
        if objective == "RMSEWithUncertainty":
            result = 2
        elif values.ndim == 2:
            result = values.shape[1]
        else:
            raise ValueError("Multioutput targets must be row-major[N,D].")
    return result


class Session(_multiclass.Session):
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, objective="MultiRMSE",
                 dimensions=None, classes=None, **kwargs):
        dimensions = _dimensions(targets, objective, dimensions, classes)
        super().__init__(bins, targets, candidate_features, candidate_bins,
                         classes=dimensions, objective=objective, **kwargs)


def train(bins, targets, candidate_features, candidate_bins, *, objective="MultiRMSE",
          dimensions=None, classes=None, **kwargs):
    dimensions = _dimensions(targets, objective, dimensions, classes)
    return _multiclass.train(bins, targets, candidate_features, candidate_bins,
                            classes=dimensions, objective=objective, **kwargs)
