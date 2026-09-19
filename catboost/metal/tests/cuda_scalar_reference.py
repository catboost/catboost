"""Independent pointwise CUDA formulas and symmetric-tree scalar oracle.

Formula sources: cuda/targets/kernel/pointwise_targets.cu::{TPoissonTarget,
THuberTarget,TExpectileTarget}; these return negative loss derivatives and
positive curvature. cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp
::WriteSecondDerivatives uses objective curvature for Newton, original sample
weights for Gradient, and adds lambda in both cases. The common walker in
descent_helpers.cpp divides by that diagonal plus 1e-20 and accepts full steps
when step_estimator.cpp::TSkipStepEstimation implements backtracking=No.

All three new losses initialize raw predictions to zero: boost_from_average
is forbidden by private/libs/options/catboost_options.cpp::Validate. This
reference accepts an explicit bias for kernel-level analytical checks only.
Float64 arithmetic deliberately makes this a numerical oracle, not a bitwise
simulation of CUDA/Metal reductions. No backend or CPU trainer is imported.
"""

import math

import numpy as np

from cuda_extended_reference import _derivatives as original_derivatives
from cuda_extended_reference import weighted_loss as original_loss
from cuda_reference import _score_children
from cuda_auxiliary_score_reference import score_children as auxiliary_score_children


def objective_terms(targets, raw, objective, objective_param=None):
    """Return per-row (loss, negative gradient, positive Hessian), unweighted."""
    targets, raw = np.broadcast_arrays(np.asarray(targets, np.float64), np.asarray(raw, np.float64))
    residual = targets - raw
    if objective == "Poisson":
        exponential = np.exp(raw)
        return exponential - targets * raw, targets - exponential, exponential
    if objective == "Huber":
        delta = float(np.float32(objective_param))
        absolute = np.abs(residual)
        inside = absolute < delta
        loss = np.where(inside, residual * residual / 2, delta * (absolute - delta / 2))
        return loss, np.clip(residual, -delta, delta), inside.astype(np.float64)
    if objective == "Expectile":
        alpha = float(np.float32(objective_param))
        multiplier = np.where(residual > 0, alpha, 1 - alpha)
        return multiplier * residual ** 2, 2 * multiplier * residual, 2 * multiplier
    if objective == "Lq":
        q = float(np.float32(objective_param))
        absolute = np.abs(residual)
        sign = np.where(residual > 0, 1, -1)
        hessian = q * (q - 1) * absolute ** (q - 2) if q >= 2 else np.ones(residual.shape)
        return absolute ** q, q * sign * absolute ** (q - 1), hessian
    if objective == "Tweedie":
        # Intentionally forward the requested variance_power. The inspected
        # CUDA host path stores it separately but accidentally forwards Alpha.
        power = float(np.float32(objective_param))
        first = targets * np.exp((1 - power) * raw)
        second = np.exp((2 - power) * raw)
        return (-first / (1 - power) + second / (2 - power), first - second,
                (power - 1) * first + (2 - power) * second)
    if objective in ("Quantile", "MAE", "LogLinQuantile"):
        alpha = (0.5 if objective == "MAE" or objective_param is None
                 else float(np.float32(objective_param)))
        scale = np.exp(raw) if objective == "LogLinQuantile" else np.ones(raw.shape)
        mismatch = targets - scale if objective == "LogLinQuantile" else residual
        multiplier = np.where(mismatch > 0, alpha, -(1 - alpha))
        return multiplier * mismatch, multiplier * scale, np.zeros(raw.shape)
    if objective == "MAPE":
        denominator = np.maximum(1, np.abs(targets))
        return (np.abs(residual) / denominator,
                np.where(residual > 0, 1, -1) / denominator, np.zeros(raw.shape))
    if objective in ("RMSE", "Logloss", "CrossEntropy"):
        gradients, hessians = original_derivatives(targets, raw, np.ones(targets.shape), objective)
        if objective == "RMSE":
            loss = residual ** 2
        else:
            encoded = (targets > 0.5).astype(float) if objective == "Logloss" else targets
            loss = encoded * np.logaddexp(0, -raw) + (1 - encoded) * np.logaddexp(0, raw)
        return loss, gradients, hessians
    raise ValueError("unsupported reference objective")


def weighted_loss(targets, raw, weights, objective, objective_param=None):
    """Poisson can have a negative mean loss; other supported losses cannot."""
    if objective in ("RMSE", "Logloss", "CrossEntropy"):
        return original_loss(targets, raw, weights, objective)
    active = weights > 0
    loss, _, _ = objective_terms(targets[active], raw[active], objective, objective_param)
    if objective == "MAE":
        # GPU MAE trains with Quantile alpha=.5 but reports twice that loss.
        loss = 2 * loss
    return float(np.dot(weights[active], loss) / weights.sum())


def exact_leaf_value(residuals, weights, objective, alpha=0.5, *, targets=None):
    """Corrected CUDA Exact selection, using original-target MAPE weights.

    Full float32 ordering replaces radix bits10..32; convergent selection
    replaces the 16-step binary search and removes its absolute FLT_EPSILON.
    Empty/zero-total leaves return zero. Zero-weight observations remain in
    the ordering, so alpha0 retains CUDA's minimum-observation convention.
    """
    values = np.asarray(residuals, np.float32)
    effective = np.asarray(weights, np.float32)
    if not values.size:
        return 0.0
    if objective == "MAPE":
        if targets is None:
            raise ValueError("MAPE Exact requires original targets for objective-correct weights")
        effective = effective / np.maximum(np.float32(1), np.abs(np.asarray(targets, np.float32)))
    effective = effective.astype(np.float64)
    if not np.any(effective > 0):
        return 0.0
    order = np.argsort(values, kind="stable")
    values, effective = values[order], effective[order]
    alpha = float(np.float32(alpha)) if objective == "Quantile" else 0.5
    if alpha <= 0:
        return float(values[0])
    if alpha >= 1:
        return float(values[np.flatnonzero(effective > 0)[-1]])
    threshold = alpha * math.fsum(effective)
    cumulative = 0.0
    for value, weight in zip(values, effective):
        cumulative += weight
        if cumulative >= threshold:
            return float(value)
    return float(values[-1])


def train_reference(bins, targets, candidate_features, candidate_bins, *, iterations,
                    depth, learning_rate, l2_leaf_reg, bias=0, score_function="Cosine",
                    objective="Poisson", objective_param=None, sample_weight=None,
                    leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                    leaf_estimation_backtracking="No"):
    """Train numeric symmetric trees with independent scalar candidate scores.

    Numeric routing uses bin > candidate threshold, with first split as bit 0.
    Every candidate remains eligible; a previously used winning split stops
    growth before being added again. Leaf estimation and structure scoring
    are independent: NewtonL2/NewtonCosine use weighted objective Hessians for
    structure denominators; L2/Cosine/SolarL2/LOOL2/SatL2 use original observation
    weights. SolarL2, LOOL2 and SatL2 omit score regularization and random noise.
    """
    if leaf_estimation_method not in ("Newton", "Gradient", "Exact"):
        raise ValueError("unsupported leaf estimation method")
    if leaf_estimation_method == "Exact" and objective not in ("Quantile", "MAE", "MAPE"):
        raise ValueError("Exact requires a quantile/median loss")
    if leaf_estimation_backtracking != "No" and leaf_estimation_method != "Exact":
        raise ValueError("this reference implements full leaf steps only")
    targets = np.asarray(targets, np.float32).astype(np.float64)
    bins = np.asarray(bins, np.uint8)
    weights = (np.ones(targets.size) if sample_weight is None
               else np.asarray(sample_weight, np.float32).astype(np.float64))
    candidates = [(int(f), int(b)) for f, b in zip(candidate_features, candidate_bins)]
    regularization = float(np.float32(l2_leaf_reg)) or float(np.float32(1e-20))
    rate = float(np.float32(learning_rate))
    prediction = np.full(targets.size, float(np.float32(bias)), np.float64)
    result = {"depths": np.zeros(iterations, np.uint32),
              "split_features": np.zeros((iterations, depth), np.uint32),
              "split_bins": np.zeros((iterations, depth), np.uint32),
              "leaf_values": np.zeros((iterations, 1 << depth)),
              "leaf_weights": np.zeros((iterations, 1 << depth)),
              "loss": np.zeros(iterations + 1)}
    result["loss"][0] = weighted_loss(targets, prediction, weights, objective, objective_param)
    for tree in range(iterations):
        _, gradients, hessians = objective_terms(targets, prediction, objective, objective_param)
        gradients *= weights
        structure_weights = weights * hessians if score_function.startswith("Newton") else weights
        score_calculator = score_function.removeprefix("Newton")
        leaf_ids = np.zeros(targets.size, np.int64)
        chosen = set()
        for level in range(depth):
            if not candidates:
                break
            best_score, winner = math.inf, None
            for candidate in candidates:
                feature, border = candidate
                child_sums, child_weights = [], []
                for leaf in range(1 << level):
                    parent = leaf_ids == leaf
                    left = parent & (bins[feature] <= border)
                    total_weight = structure_weights[parent].sum()
                    total_sum = gradients[parent].sum()
                    left_weight = structure_weights[left].sum()
                    left_sum = gradients[left].sum()
                    child_sums.extend([left_sum, total_sum - left_sum])
                    child_weights.extend([left_weight, max(total_weight - left_weight, 0)])
                if score_calculator in ("SolarL2", "LOOL2", "SatL2"):
                    score = auxiliary_score_children(child_sums, child_weights, score_calculator)
                else:
                    score = _score_children(np.asarray(child_sums), np.asarray(child_weights),
                                            regularization, score_calculator)
                if score < best_score:
                    best_score, winner = score, candidate
            if winner in chosen:
                break
            if winner is None:
                raise ArithmeticError("no finite candidate score")
            chosen.add(winner)
            feature, border = winner
            result["split_features"][tree, level] = feature
            result["split_bins"][tree, level] = border
            result["depths"][tree] += 1
            leaf_ids |= (bins[feature] > border).astype(np.int64) << level
        count = 1 << int(result["depths"][tree])
        leaf_weights = np.array([weights[leaf_ids == leaf].sum() for leaf in range(count)])
        point = np.zeros(count)
        if leaf_estimation_method == "Exact":
            residual = targets.astype(np.float32) - prediction.astype(np.float32)
            for leaf in range(count):
                members = leaf_ids == leaf
                point[leaf] = exact_leaf_value(residual[members], weights[members], objective,
                                               0.5 if objective_param is None else objective_param,
                                               targets=targets[members])
        for _ in range(0 if leaf_estimation_method == "Exact" else leaf_estimation_iterations):
            _, derivatives, hessians = objective_terms(targets, prediction + point[leaf_ids],
                                                      objective, objective_param)
            for leaf in range(count):
                members = (leaf_ids == leaf) & (weights > 0)
                gradient = np.dot(weights[members], derivatives[members])
                curvature = (np.dot(weights[members], hessians[members])
                             if leaf_estimation_method == "Newton" else leaf_weights[leaf])
                if leaf_weights[leaf] < 1e-20:
                    point[leaf] = 0
                elif curvature + regularization > 0:
                    point[leaf] += gradient / (curvature + regularization + 1e-20)
        result["leaf_values"][tree, :count] = point * rate
        result["leaf_weights"][tree, :count] = leaf_weights
        prediction += rate * point[leaf_ids]
        result["loss"][tree + 1] = weighted_loss(targets, prediction, weights, objective, objective_param)
    result["predictions"] = prediction
    result["rmse"] = result["loss"]
    return result
