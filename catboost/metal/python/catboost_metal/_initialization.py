"""Host initialization for objectives that support boost_from_average.

Quantile/MAE/MAPE follow libs/metrics/optimal_const_for_loss.h, retaining its
delta convention. MAE parameters here are private helper behavior: the public
loss parser accepts no MAE parameters, matching TQuantileMetric::Create in
libs/metrics/metric.cpp. Public MAE therefore always uses delta=1e-6.
Full value ordering replaces CalcSampleQuantile's bounded
value-space bisection, which can return a non-target on wide-range inputs.
Weighted selection uses the actual threshold, without an absolute epsilon.
"""

import math

import numpy as np


def _weighted_quantile(values, weights, alpha):
    """Select an observed value, with no interpolation or weight-size epsilon."""
    order = np.argsort(values, kind="stable")
    ordered_values, ordered_weights = values[order], weights[order]
    if alpha <= 0:
        return float(ordered_values[0])
    if alpha >= 1:
        return float(ordered_values[np.flatnonzero(ordered_weights > 0)[-1]])
    total = math.fsum(ordered_weights)
    cumulative = np.cumsum(ordered_weights, dtype=np.float64)
    selected = min(int(np.searchsorted(cumulative, alpha * total, side="left")), len(values) - 1)
    return float(ordered_values[selected])


def initialize_bias(targets, weights, objective, parameters):
    """Return a float32-representable scalar bias for encoded numeric targets.

    ``weights`` may be None; ``parameters`` contains parsed loss parameters.
    Caller invokes this only when boost_from_average is enabled. Binary labels
    are encoded as 0/1 for Logloss; CrossEntropy permits fractional labels.
    RMSE/binary means retain the frontend's double-product accumulation to
    avoid overflow in otherwise representable weighted averages.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        target = np.asarray(targets, dtype=np.float32)
        weight = (np.ones(target.shape, dtype=np.float32) if weights is None
                  else np.asarray(weights, dtype=np.float32))
    if (target.ndim != 1 or not target.size or weight.shape != target.shape
            or not np.isfinite(target).all() or not np.isfinite(weight).all()
            or (weight < 0).any() or not (weight > 0).any()):
        raise ValueError("Bias initialization requires finite targets and nonnegative weights with positive total.")
    target64, weight64 = target.astype(np.float64), weight.astype(np.float64)
    if objective in ("RMSE", "Logloss", "CrossEntropy"):
        mean = float(np.float32(np.dot(target64, weight64) / math.fsum(weight64)))
        if objective == "RMSE":
            return mean
        if not 0 < mean < 1:
            raise ValueError("boost_from_average requires a weighted binary mean strictly between 0 and 1.")
        return float(np.float32(math.log(mean / (1 - mean))))
    if objective == "MAPE":
        # MAPE's true objective denominator uses original targets. Metal also
        # uses it for Exact leaves, correcting CUDA's residual-weight defect.
        adjusted = (weight / np.maximum(np.float32(1), np.abs(target))).astype(np.float64)
        if not (adjusted > 0).any():
            raise ValueError("MAPE initialization weights underflowed to zero; rescale targets or weights.")
        return float(np.float32(_weighted_quantile(target64, adjusted, 0.5)))
    if objective not in ("MAE", "Quantile"):
        raise ValueError(f"boost_from_average is not supported for {objective}.")
    alpha = 0.5 if objective == "MAE" else float(parameters.get("alpha", 0.5))
    delta = float(parameters.get("delta", 1e-6))
    if not np.isfinite([alpha, delta]).all() or not 0 <= alpha <= 1 or not 0 <= delta <= 0.01:
        raise ValueError("Quantile initialization requires alpha in [0,1] and delta in [0,0.01].")
    quantile = _weighted_quantile(target64, weight64, alpha)
    if delta > 0:
        below = math.fsum(weight64[target64 < quantile])
        equal = math.fsum(weight64[target64 == quantile])
        # Preserve CUDA's host delta adjustment, including its DBL_EPSILON
        # comparison. The weighted quantile selection itself has no epsilon.
        if below + equal * alpha >= alpha * math.fsum(weight64) - np.finfo(np.float64).eps:
            quantile -= delta
        else:
            quantile += delta
    return float(np.float32(quantile))
