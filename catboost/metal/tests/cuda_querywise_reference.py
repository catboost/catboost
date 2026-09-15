"""Independent scalar CUDA query objectives, split search, and leaf walker.

Sources: cuda/targets/kernel/query_{rmse,softmax}.cu, querywise_targets_impl.h,
and cuda/methods/leaves_estimation/{descent_helpers,step_estimator}.cpp.
Queries always use original row boundaries and original observation weights;
leaf projection and structure bootstrap occur only after query normalization.
No Metal optimizer or CPU trainer is imported.
"""

import numpy as np

from cuda_reference import _score_children
from cuda_scalar_reference import auxiliary_score_children


def query_terms(targets, predictions, weights, offsets, objective, beta=1, query_lambda=0.01):
    targets, predictions, weights = (np.asarray(values, np.float64) for values in (targets, predictions, weights))
    beta, query_lambda = float(np.float32(beta)), float(np.float32(query_lambda))
    gradients, hessians = np.zeros(targets.size), np.zeros(targets.size)
    numerator = denominator = 0.0
    for start, end in zip(offsets[:-1], offsets[1:]):
        indices = np.arange(int(start), int(end))
        active = indices[weights[indices] > 0]
        w, y, raw = weights[active], targets[active], predictions[active]
        if not active.size:
            continue
        if objective == "QueryRMSE":
            residual = y - raw
            mean = np.dot(w, residual) / w.sum()
            centered = residual - mean
            gradients[active] = w * centered
            # CUDA uses w, not the exact coupled Hessian's w*(1-w/sum(w)).
            hessians[active] = w
            numerator += np.dot(w, centered * centered)
            denominator += w.sum()
        elif objective == "QuerySoftMax":
            logits = beta * raw
            shift = np.max(logits)
            exponentials = w * np.exp(logits - shift)
            total_exp = exponentials.sum()
            probability = exponentials / total_exp
            target_mass = np.dot(w, y)
            gradients[active] = beta * (w * y - target_mass * probability)
            if target_mass > 0:
                hessians[active] = beta * target_mass * (
                    beta * probability * (1 - probability) + query_lambda)
            positive = y > 0
            log_probability = np.log(w) + logits - shift - np.log(total_exp)
            numerator -= np.dot((w * y)[positive], log_probability[positive])
            denominator += target_mass
        else:
            raise ValueError("not a querywise objective")
    return gradients, hessians, float(numerator), float(denominator)


def query_loss(targets, predictions, weights, offsets, objective, beta=1, query_lambda=0.01):
    _, _, numerator, denominator = query_terms(
        targets, predictions, weights, offsets, objective, beta, query_lambda)
    if denominator <= 0:
        raise ValueError("query metric denominator must be positive")
    mean = numerator / denominator
    return float(np.sqrt(mean) if objective == "QueryRMSE" else mean)


def structure_reference(bins, targets, predictions, weights, offsets, candidate_features,
                        candidate_bins, *, objective, depth, l2_leaf_reg, score_function,
                        query_beta=1, query_lambda=0.01, bootstrap_factors=None):
    gradients, hessians, _, _ = query_terms(targets, predictions, weights, offsets,
                                           objective, query_beta, query_lambda)
    gradients = gradients.astype(np.float32).astype(np.float64)
    structure_weights = np.asarray(weights, np.float32).astype(np.float64)
    if score_function.startswith("Newton"):
        structure_weights = hessians.astype(np.float32).astype(np.float64)
        if (structure_weights < 0).any() or not np.isfinite(structure_weights).all():
            raise ArithmeticError("Newton query structure weights must be nonnegative and finite")
    if bootstrap_factors is not None:
        gradients = (gradients * bootstrap_factors).astype(np.float32).astype(np.float64)
        structure_weights = (structure_weights * bootstrap_factors).astype(np.float32).astype(np.float64)
    regularization = float(np.float32(l2_leaf_reg)) or float(np.float32(1e-20))
    family = score_function.removeprefix("Newton")
    ids = np.zeros(targets.size, np.int64)
    splits = []
    for level in range(depth):
        scores = []
        for feature, border in zip(candidate_features, candidate_bins):
            sums, masses = [], []
            for leaf in range(1 << level):
                parent = ids == leaf
                left = parent & (bins[feature] <= border)
                right = parent & ~left
                sums.extend([gradients[left].sum(), gradients[right].sum()])
                masses.extend([structure_weights[left].sum(), structure_weights[right].sum()])
            score = (auxiliary_score_children(sums, masses, family) if family in ("SolarL2", "LOOL2")
                     else _score_children(np.asarray(sums), np.asarray(masses), regularization, family))
            scores.append(score)
        if not scores:
            break
        winner = int(np.argmin(scores))
        split = (int(candidate_features[winner]), int(candidate_bins[winner]))
        if split in splits:
            break
        splits.append(split)
        ids |= (bins[split[0]] > split[1]).astype(np.int64) << level
    return splits, ids


def leaf_reference(targets, baseline, weights, offsets, leaf_ids, leaf_count, *, objective,
                   l2_leaf_reg, leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                   leaf_estimation_backtracking="No", query_beta=1, query_lambda=0.01):
    targets, baseline, weights = (np.asarray(v, np.float32) for v in (targets, baseline, weights))
    masses = np.bincount(leaf_ids, weights=weights.astype(np.float64), minlength=leaf_count)
    regularization = float(np.float32(l2_leaf_reg)) or float(np.float32(1e-20))
    point = np.zeros(leaf_count, np.float32)
    trace = []

    def project(candidate):
        raw = (baseline + candidate[leaf_ids]).astype(np.float32)
        g, h, numerator, _ = query_terms(targets, raw, weights, offsets, objective, query_beta, query_lambda)
        with np.errstate(over="ignore", invalid="ignore"):
            g = g.astype(np.float32).astype(np.float64)
            # Gradient must not inspect unused Hessians, including overflow.
            h = np.zeros_like(g) if leaf_estimation_method == "Gradient" else h.astype(np.float32).astype(np.float64)
        return (-numerator, np.bincount(leaf_ids, weights=g, minlength=leaf_count),
                np.bincount(leaf_ids, weights=h, minlength=leaf_count))

    def move(gradient, hessian):
        diagonal = (masses if leaf_estimation_method == "Gradient" else hessian) + regularization
        populated = masses >= 1e-20
        if not np.isfinite(gradient).all() or not np.isfinite(diagonal).all():
            raise ArithmeticError("nonfinite query leaf derivatives")
        if (diagonal[populated] <= 0).any():
            raise ArithmeticError("query Newton diagonal is not positive")
        return np.divide(gradient, diagonal + 1e-20, out=np.zeros(leaf_count),
                         where=diagonal > 0).astype(np.float32)

    def candidate(direction, step):
        updated = (point.astype(np.float64) + step * direction.astype(np.float64)).astype(np.float32)
        updated[masses < 1e-20] = 0
        return updated

    value, gradient, hessian = project(point)
    if leaf_estimation_backtracking == "No" or leaf_estimation_iterations == 1:
        for _ in range(leaf_estimation_iterations):
            point = candidate(move(gradient, hessian), 1)
            value, gradient, hessian = project(point)
            trace.append((1.0, True))
        return point, masses, trace
    attempts, ever_updated = 0, False
    while attempts < leaf_estimation_iterations:
        direction = move(gradient, hessian)
        dot = np.dot(gradient, direction.astype(np.float64))
        step = 1.0
        while attempts < leaf_estimation_iterations or (not ever_updated and attempts < 100):
            trial = candidate(direction, step)
            next_value, next_gradient, next_hessian = project(trial)
            threshold = value + (1e-5 * step * dot if leaf_estimation_backtracking == "Armijo" else 0)
            accepted = np.isfinite(next_value) and next_value >= threshold
            attempts += 1
            trace.append((step, bool(accepted)))
            if accepted:
                point = trial
                value, gradient, hessian = next_value, next_gradient, next_hessian
                ever_updated = True
                break
            step *= 0.5
    return point, masses, trace


def train_reference(bins, targets, features, borders, *, iterations, depth, learning_rate,
                    l2_leaf_reg, bias, objective, group_offsets, score_function="Cosine",
                    sample_weight=None, initial_predictions=None, query_beta=1, query_lambda=0.01,
                    leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                    leaf_estimation_backtracking="No", bootstrap_factors=None):
    targets = np.asarray(targets, np.float32)
    weights = np.ones(targets.size, np.float32) if sample_weight is None else np.asarray(sample_weight, np.float32)
    cursor = (np.full(targets.size, np.float32(bias), np.float32) if initial_predictions is None
              else np.asarray(initial_predictions, np.float32).copy())
    result = dict(depths=np.zeros(iterations, np.uint32),
                  split_features=np.zeros((iterations, depth), np.uint32),
                  split_bins=np.zeros((iterations, depth), np.uint32),
                  leaf_values=np.zeros((iterations, 1 << depth), np.float32),
                  leaf_weights=np.zeros((iterations, 1 << depth)), loss=np.zeros(iterations + 1))
    result["loss"][0] = query_loss(targets, cursor, weights, group_offsets, objective, query_beta, query_lambda)
    for iteration in range(iterations):
        factors = None if bootstrap_factors is None else bootstrap_factors[iteration]
        splits, ids = structure_reference(
            bins, targets, cursor, weights, group_offsets, features, borders, objective=objective,
            depth=depth, l2_leaf_reg=l2_leaf_reg, score_function=score_function,
            query_beta=query_beta, query_lambda=query_lambda, bootstrap_factors=factors)
        point, masses, _ = leaf_reference(
            targets, cursor, weights, group_offsets, ids, 1 << len(splits), objective=objective,
            l2_leaf_reg=l2_leaf_reg, leaf_estimation_method=leaf_estimation_method,
            leaf_estimation_iterations=leaf_estimation_iterations,
            leaf_estimation_backtracking=leaf_estimation_backtracking,
            query_beta=query_beta, query_lambda=query_lambda)
        result["depths"][iteration] = len(splits)
        for level, (feature, border) in enumerate(splits):
            result["split_features"][iteration, level] = feature
            result["split_bins"][iteration, level] = border
        values = (point * np.float32(learning_rate)).astype(np.float32)
        result["leaf_values"][iteration, :values.size] = values
        result["leaf_weights"][iteration, :values.size] = masses
        cursor = (cursor + values[ids]).astype(np.float32)
        result["loss"][iteration + 1] = query_loss(targets, cursor, weights, group_offsets,
                                                 objective, query_beta, query_lambda)
    result["predictions"] = cursor
    return result
