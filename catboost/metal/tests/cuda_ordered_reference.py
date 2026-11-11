"""Independent numeric Ordered boosting oracle, derived from CatBoost CUDA.

This is deliberately a small, slow equation interpreter, not a fallback
trainer. It does not load Metal or fit a CPU CatBoost model. Sources:

* cuda/methods/dynamic_boosting.h:176-232,250-473: folds, search permutation,
  independent prefix cursors, prefix/full-model estimation and shrinkage.
* cuda/methods/kernel/pointwise_scores.cu:318-418: dynamic Cosine scoring.
* cuda/methods/oblivious_tree_structure_searcher.cpp:257-267: repeated-split stop.
* cuda/targets/kernel/pointwise_targets.cu:11-217: scalar objective formulas;
  CrossEntropyImpl supplies weighted binary derivatives.
* cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp:140-165,
  253-261: Newton/Gradient denominators and per-task normalization.
* cuda/methods/leaves_estimation/descent_helpers.cpp: diagonal leaf steps.

Only permutation generation and selection reuse independently tested _ordered_rng helpers. Fold geometry,
derivatives, candidate reductions, scoring and leaf updates are independent.
Inputs/configuration are narrowed to float32, then arithmetic/reductions use
float64; the CUDA sigmoid's float32 saturation is retained. Thus comparisons
should allow rounding error, rather than require identical reductions or trees
for exactly/nearly tied candidate scores. Bootstrap and score noise are absent.
"""

import copy
import math

import numpy as np

from catboost_metal._ordered_rng import OrderedSelectionRng, cuda_ordered_history_order


def numeric_folds(rows, fold_len_multiplier=2.0, min_fold_size=100):
    """Return CUDA numeric (estimate_end, quality_end) exclusive boundaries.

    Numeric NextQueryOffsetForLine(x) advances to min(x+1,N), including the
    initial minimum. This differs from merely doubling a prefix of one row.
    """
    if not isinstance(rows, (int, np.integer)) or not 4 <= rows <= 1 << 24:
        raise ValueError("numeric Ordered requires 4..2**24 rows")
    if not isinstance(min_fold_size, (int, np.integer)) or min_fold_size < 1:
        raise ValueError("min_fold_size must be a positive integer")
    growth = float(np.float32(fold_len_multiplier))
    if not math.isfinite(growth) or growth <= 1:
        raise ValueError("fold_len_multiplier must be finite and greater than one")
    rows, minimum = int(rows), int(min_fold_size)
    if rows < 500:
        first = 1
    else:
        ratio = (rows + minimum - 1) // minimum
        # ceil(log2(ratio)) calculated exactly without floating boundary issues.
        count = (ratio - 1).bit_length()
        first = ((rows + (1 << 18) - 1) // (1 << 18)
                 if count >= 18 else min(minimum, rows // 50))
    prefix = min(first + 1, rows)
    result = []
    while prefix < rows:
        end = min(math.floor(prefix * growth) + 1, rows)
        result.append((prefix, end))
        if len(result) > 4096:
            raise ValueError("numeric Ordered exceeds the 4096-fold limit")
        prefix = end
    return np.asarray(result, dtype=np.uint32).reshape(-1, 2)


def _sigmoid(raw):
    probability = np.empty_like(raw, dtype=np.float64)
    positive = raw >= 0
    probability[positive] = 1 / (1 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    probability[~positive] = exponential / (1 + exponential)
    return np.maximum(probability.astype(np.float32), np.float32(1e-40)).astype(np.float64)


def objective_terms(labels, cursor, objective, objective_param=None):
    """Unweighted (reported loss, negative gradient, positive curvature).

    These equations are independently spelled out from pointwise_targets.cu.
    Equality takes CUDA's negative subgradient for quantile/MAPE/Lq(q=1).
    Tweedie consumes the requested variance power, correcting the upstream
    host path that accidentally forwards its separate Alpha field. MAE's
    reported loss is full absolute error; its training gradient remains half.
    """
    labels, cursor = np.broadcast_arrays(np.asarray(labels, np.float64), np.asarray(cursor, np.float64))
    residual = labels - cursor
    if objective == "RMSE":
        return residual ** 2, residual, np.ones(residual.shape)
    if objective in ("Logloss", "CrossEntropy"):
        probability = _sigmoid(cursor)
        loss = labels * np.logaddexp(0, -cursor) + (1 - labels) * np.logaddexp(0, cursor)
        return loss, labels - probability, probability * (1 - probability)
    if objective == "Poisson":
        exponential = np.exp(cursor)
        return exponential - labels * cursor, labels - exponential, exponential
    if objective == "Huber":
        delta = float(np.float32(objective_param))
        absolute = np.abs(residual)
        interior = absolute < delta
        loss = np.where(interior, .5 * residual ** 2, delta * (absolute - .5 * delta))
        return loss, np.clip(residual, -delta, delta), interior.astype(np.float64)
    if objective == "Expectile":
        alpha = float(np.float32(objective_param))
        multiplier = np.where(residual > 0, alpha, 1 - alpha)
        return multiplier * residual ** 2, 2 * multiplier * residual, 2 * multiplier
    if objective == "Lq":
        power = float(np.float32(objective_param))
        absolute = np.abs(residual)
        gradient = power * np.where(residual > 0, 1, -1) * absolute ** (power - 1)
        curvature = (power * (power - 1) * absolute ** (power - 2)
                     if power >= 2 else np.ones(residual.shape))
        return absolute ** power, gradient, curvature
    if objective == "Tweedie":
        variance = float(np.float32(objective_param))
        first = np.zeros(residual.shape)
        positive = labels != 0
        first[positive] = labels[positive] * np.exp((1 - variance) * cursor[positive])
        second = np.exp((2 - variance) * cursor)
        return (-first / (1 - variance) + second / (2 - variance),
                first - second, (variance - 1) * first + (2 - variance) * second)
    if objective in ("LogLinQuantile", "Quantile", "MAE"):
        alpha = .5 if objective == "MAE" or objective_param is None else float(np.float32(objective_param))
        prediction = np.exp(cursor) if objective == "LogLinQuantile" else cursor
        mismatch = labels - prediction
        multiplier = np.where(mismatch > 0, alpha, -(1 - alpha))
        gradient = multiplier * prediction if objective == "LogLinQuantile" else multiplier
        loss = np.abs(mismatch) if objective == "MAE" else multiplier * mismatch
        return loss, gradient, np.zeros(residual.shape)
    if objective == "MAPE":
        denominator = np.maximum(1, np.abs(labels))
        return np.abs(residual) / denominator, np.where(residual > 0, 1, -1) / denominator, np.zeros(residual.shape)
    raise ValueError("unsupported Ordered oracle objective")


def _derivatives(labels, cursor, weights, objective, objective_param=None):
    # The CUDA-derived Metal helper skips inactive rows before evaluating any
    # exponential/power, so an unused zero-weight outlier cannot overflow.
    active = weights > 0
    gradient, hessian = np.zeros(weights.shape), np.zeros(weights.shape)
    _, derivative, curvature = objective_terms(labels[active], cursor[active], objective, objective_param)
    gradient[active], hessian[active] = weights[active] * derivative, weights[active] * curvature
    return gradient, hessian


def exact_leaf_value(targets, cursor, weights, objective, alpha=.5):
    """Corrected CUDA weighted residual quantile for one independent task.

    Full float32 residual ordering replaces upstream truncated radix keys;
    lower-bound selection has no fixed search-count or absolute epsilon shift.
    MAPE uses original target magnitude for its weights, because residual-based
    weights optimize a different objective when a task cursor is nonzero.
    """
    residuals = np.asarray(targets, np.float32) - np.asarray(cursor, np.float32)
    effective = np.asarray(weights, np.float32)
    if objective == "MAPE":
        effective = effective / np.maximum(np.float32(1), np.abs(np.asarray(targets, np.float32)))
    effective = effective.astype(np.float64)
    if not residuals.size or not np.any(effective > 0):
        return 0.0
    order = np.argsort(residuals, kind="stable")
    residuals, effective = residuals[order], effective[order]
    alpha = float(np.float32(alpha)) if objective == "Quantile" else .5
    if alpha == 0:
        return float(residuals[0])
    if alpha == 1:
        return float(residuals[np.flatnonzero(effective > 0)[-1]])
    threshold, cumulative = alpha * math.fsum(effective), 0.0
    for residual, weight in zip(residuals, effective):
        cumulative += weight
        if cumulative >= threshold:
            return float(residual)
    return float(residuals[-1])


def estimate_tasks_reference(tasks, leaves, *, objective, objective_param=None,
                             leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                             leaf_estimation_backtracking="No", l2_leaf_reg=3,
                             fold_size_loss_normalization=False):
    """One CUDA walker over all task leaf vectors, with one common step size.

    Each task is (targets, initial_prefix_cursor, original_weights, leaf_ids).
    CUDA's TObliviousTreeLeavesEstimator combines every task into one point and
    sums negative objective values; the TNewtonLikeWalker accepts or rejects
    all tasks together. Rejected trials consume the iteration budget. Before
    any successful update, at most 100 total trials permit a safe first step.
    The returned trace exposes global and per-task trial values for checking
    that independent per-prefix line searches would change this algorithm.
    """
    if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
        raise ValueError("unsupported backtracking rule")
    l2 = float(np.float32(l2_leaf_reg)) or float(np.float32(1e-20))
    point = np.zeros((len(tasks), leaves), np.float32)
    mass = np.asarray([np.bincount(task[3], weights=task[2], minlength=leaves) for task in tasks])
    trace = []

    def evaluate(values):
        gradient, diagonal = np.zeros(point.shape), np.zeros(point.shape)
        scores = np.zeros(len(tasks))
        for index, (labels, baseline, weights, leaf_ids) in enumerate(tasks):
            active = weights > 0
            if not active.any():
                continue
            labels = np.asarray(labels)[active]
            raw = np.asarray(baseline)[active] + values[index, leaf_ids[active]].astype(np.float64)
            selected_weights = weights[active]
            loss, first, second = objective_terms(labels, raw, objective, objective_param)
            if objective == "MAE":
                loss *= .5  # CUDA training Score is Quantile(.5), report is MAE.
            denominator = selected_weights.sum() if fold_size_loss_normalization else 1.0
            scores[index] = -np.dot(selected_weights, loss) / denominator
            gradient[index] = np.bincount(leaf_ids[active], weights=selected_weights * first,
                                          minlength=leaves) / denominator
            diagonal[index] = (np.bincount(leaf_ids[active], weights=selected_weights * second,
                                           minlength=leaves) if leaf_estimation_method == "Newton"
                               else mass[index]) / denominator
        return scores, gradient, diagonal + l2

    task_score, gradient, diagonal = evaluate(point)
    attempts, updated = 0, False
    while attempts < leaf_estimation_iterations:
        direction = np.divide(gradient, diagonal + 1e-20, out=np.zeros(point.shape),
                               where=diagonal > 0).astype(np.float32)
        current_score = float(task_score.sum())
        dot = float(np.sum(gradient * direction.astype(np.float64)))
        step = 1.0
        while attempts < leaf_estimation_iterations or (not updated and attempts < 100):
            trial = (point.astype(np.float64) + step * direction.astype(np.float64)).astype(np.float32)
            trial[mass < 1e-20] = 0
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                next_task_score, next_gradient, next_diagonal = evaluate(trial)
            score = float(next_task_score.sum())
            threshold = current_score + (1e-5 * step * dot if leaf_estimation_backtracking == "Armijo" else 0)
            accepted = (leaf_estimation_iterations == 1 or leaf_estimation_backtracking == "No"
                        or score >= threshold)
            trace.append({"step": step, "accepted": accepted, "score_before": current_score,
                          "score_after": score, "direction_dot": dot,
                          "task_scores_before": task_score.copy(), "task_scores_after": next_task_score.copy()})
            attempts += 1
            if accepted:
                point, task_score, gradient, diagonal = trial, next_task_score, next_gradient, next_diagonal
                updated = True
                break
            step *= .5
    return point.astype(np.float64), mass, trace


def weighted_loss(targets, predictions, weights, objective, objective_param=None):
    """Reported full-model training metric, normalized by original weight."""
    active = weights > 0
    terms, _, _ = objective_terms(targets[active], predictions[active], objective, objective_param)
    value = float(np.dot(weights[active], terms) / weights.sum())
    return math.sqrt(value) if objective == "RMSE" else value


def _task_descriptors(boundaries, rows, permutation_count):
    learning_count = permutation_count - 1 if permutation_count > 1 else 1
    descriptors = []
    offset = 0
    for permutation in range(learning_count):
        for prefix, end in boundaries:
            descriptors.append((int(prefix), int(end), offset, permutation))
            offset += int(end)
    descriptors.append((rows, rows, offset, permutation_count - 1))
    return np.asarray(descriptors, dtype=np.uint32), offset + rows


def _score_candidate(bins, feature, border, partitions, fold_derivatives,
                     descriptors, permutations, l2, normalize, leaves, one_hot=False):
    numerator, norm = 0.0, 1e-20
    for descriptor, (gradients, denominators) in zip(descriptors, fold_derivatives):
        prefix, end, _, permutation_id = map(int, descriptor)
        order = permutations[permutation_id, :end]
        # Interleaved child IDs are only reduction indices. Model leaf IDs
        # below retain the CatBoost little-endian split-bit convention.
        children = 2 * partitions[order] + (bins[feature, order] == border if one_hot else bins[feature, order] > border)
        estimate_weights = np.bincount(children[:prefix], weights=denominators[:prefix],
                                       minlength=2 * leaves)
        estimate_gradient = np.bincount(children[:prefix], weights=gradients[:prefix],
                                        minlength=2 * leaves)
        quality_weights = np.bincount(children[prefix:], weights=denominators[prefix:],
                                      minlength=2 * leaves)
        quality_gradient = np.bincount(children[prefix:], weights=gradients[prefix:],
                                       minlength=2 * leaves)
        ridge = l2 * estimate_weights if normalize else l2
        mu = np.divide(estimate_gradient, estimate_weights + ridge,
                       out=np.zeros(2 * leaves), where=estimate_weights > 0)
        numerator += float(np.dot(quality_gradient, mu))
        norm += float(np.dot(quality_weights, mu * mu))
    return -numerator / math.sqrt(norm) if norm > 1e-15 else float(np.finfo(np.float32).max)


def train_reference(
    bins, targets, candidate_features, candidate_bins, *, iterations, depth,
    learning_rate, l2_leaf_reg, bias, objective="RMSE", objective_param=None, score_function="Cosine",
    sample_weight=None, leaf_estimation_method="Newton", leaf_estimation_iterations=1,
    leaf_estimation_backtracking="No",
    initial_predictions=None, permutation_count=1, permutations=None,
    fold_len_multiplier=2.0, min_fold_size=100, fold_size_loss_normalization=False, fold_permutation_block=64,
    random_seed=0, iteration_offset=0, initial_state=None, candidate_types=None, group_sizes=None, permutation_bins=None, ctr_unique_values=None, model_size_reg=.5, feature_weights=None,
):
    """Train new numeric Ordered trees and expose every independent cursor.

    ``bins`` is uint8 [feature, original_row]. Optional ``permutations`` is
    uint32 [P,position], containing original-row IDs. P-1 is the estimation
    permutation; max(P-1,1) learning permutations each own all numeric folds.
    Only one learning permutation selects structure, following CUDA's literal
    modulo rule. All learning permutations receive independent prefix solves.

    Result model arrays are padded to requested depth/iterations. ``folds`` is
    uint32 [task,4]: estimate end, quality end, packed cursor offset, permutation
    ID. Learning tasks come first, then the independent full-estimation task.
    ``cursors`` is the flat task-packed prediction state in permutation order;
    ``predictions`` restores the full-estimation cursor to original row order.
    ``task_leaf_values`` exposes the shrunk independent prefix vectors for
    leakage checks. ``loss`` has the initial value plus one value per new tree.

    Resume with ``initial_state=result['state']``. This restores all cursors;
    using only the exported model predictions cannot reproduce Ordered state.
    A nonzero explicit iteration_offset overrides the state's completed count.
    """
    if objective not in ("RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber", "Expectile",
                         "Lq", "Tweedie", "LogLinQuantile", "Quantile", "MAE", "MAPE"):
        raise ValueError("unsupported Ordered oracle objective")
    if score_function not in ("Cosine", "NewtonCosine"):
        raise ValueError("Ordered oracle score_function must be Cosine or NewtonCosine")
    if leaf_estimation_method not in ("Newton", "Gradient", "Exact"):
        raise ValueError("Ordered oracle leaf_estimation_method must be Newton, Gradient or Exact")
    if leaf_estimation_method == "Exact" and objective not in ("Quantile", "MAE", "MAPE"):
        raise ValueError("Exact leaf estimation requires Quantile, MAE or MAPE")
    if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
        raise ValueError("unsupported Ordered oracle backtracking rule")
    if objective in ("Huber", "Expectile", "Lq", "Tweedie") and objective_param is None:
        raise ValueError(f"{objective} requires objective_param")
    if objective_param is not None:
        objective_param = float(np.float32(objective_param))
        if not math.isfinite(objective_param):
            raise ValueError("objective_param must be finite")
    if objective == "Huber" and objective_param < 0:
        raise ValueError("Huber delta must be nonnegative")
    if objective in ("Expectile", "LogLinQuantile", "Quantile"):
        if objective_param is None:
            objective_param = .5
        if not 0 <= objective_param <= 1:
            raise ValueError("alpha must be in [0,1]")
    if objective == "Lq" and objective_param < 1:
        raise ValueError("Lq q must be at least one")
    if objective == "Tweedie" and not 1 < objective_param < 2:
        raise ValueError("Tweedie variance power must lie strictly between one and two")
    if leaf_estimation_method == "Newton" and (
        objective in ("LogLinQuantile", "Quantile", "MAE", "MAPE")
        or (objective == "Lq" and objective_param < 2)
    ):
        raise ValueError("this objective requires Gradient leaf estimation")
    if not isinstance(iterations, (int, np.integer)) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not isinstance(depth, (int, np.integer)) or not 0 <= depth <= 16:
        raise ValueError("depth must be an integer in [0,16]")
    if not isinstance(leaf_estimation_iterations, (int, np.integer)) or leaf_estimation_iterations < 1:
        raise ValueError("leaf_estimation_iterations must be positive")
    bins = np.asarray(bins, dtype=np.uint8)
    labels = np.asarray(targets, dtype=np.float32).astype(np.float64)
    if labels.ndim != 1 or not np.isfinite(labels).all():
        raise ValueError("targets must be a finite vector")
    rows = labels.size
    if bins.ndim != 2 or bins.shape[1] != rows:
        raise ValueError("bins must be feature-major with one column per target")
    if objective == "Logloss":
        labels = (labels > 0.5).astype(np.float64)
    elif objective == "CrossEntropy" and ((labels < 0).any() or (labels > 1).any()):
        raise ValueError("CrossEntropy targets must be in [0,1]")
    if objective in ("Poisson", "Tweedie") and (labels < 0).any():
        raise ValueError(f"{objective} targets must be nonnegative")
    weights = (np.ones(rows, dtype=np.float64) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float32).astype(np.float64))
    if (weights.shape != labels.shape or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
        raise ValueError("weights must be finite, nonnegative and have positive total")
    features = np.asarray(candidate_features, dtype=np.uint32)
    borders = np.asarray(candidate_bins, dtype=np.uint32)
    types = np.zeros(len(features), np.uint8) if candidate_types is None else np.asarray(candidate_types, np.uint8)
    if types.shape != features.shape or (types > 1).any():
        raise ValueError("candidate_types must contain one comparison per candidate")
    if (features.ndim != 1 or features.shape != borders.shape
            or (features >= bins.shape[0]).any() or (borders > 255).any()):
        raise ValueError("candidate vectors must match and address valid bins")
    if not isinstance(permutation_count, (int, np.integer)) or not 1 <= permutation_count <= 64:
        raise ValueError("permutation_count must be an integer in [1,64]")
    permutation_count = int(permutation_count)
    counts = np.zeros(bins.shape[0], np.uint32) if ctr_unique_values is None else np.asarray(ctr_unique_values, np.uint32)
    feature_weights = np.ones(bins.shape[0]) if feature_weights is None else np.asarray(feature_weights, np.float32)
    maximum = max(1, int(counts.max(initial=0)))
    penalties = np.array([np.float32(math.pow(float(np.float32(1 + np.float32(c) / np.float32(maximum))),
        -float(np.float32(model_size_reg)))) if c else 1. for c in counts])
    banks = (np.repeat(bins[None], permutation_count, axis=0) if permutation_bins is None
             else np.asarray(permutation_bins, dtype=np.uint8))
    if banks.shape != (permutation_count, *bins.shape) or not np.array_equal(banks[0], bins):
        raise ValueError("feature banks must match every permutation with bins as bank zero")
    if group_sizes is not None:
        from cuda_ordered_group_reference import checked_group_sizes, reference_group_order, reference_group_folds
        group_sizes = checked_group_sizes(group_sizes)
        if sum(group_sizes) != rows:
            raise ValueError("Group sizes must cover all rows")
    if permutations is None:
        permutations = np.stack([reference_group_order(group_sizes, p, fold_permutation_block)[1]
            if group_sizes is not None else cuda_ordered_history_order(rows, p, fold_permutation_block)
            for p in range(permutation_count)])
    else:
        permutations = np.asarray(permutations, dtype=np.uint32)
    if (permutations.shape != (permutation_count, rows)
            or not np.all(np.sort(permutations, axis=1) == np.arange(rows))):
        raise ValueError("each permutation must contain every original row once")
    if group_sizes is None:
        boundaries = numeric_folds(rows, fold_len_multiplier, min_fold_size)
        descriptors, packed_rows = _task_descriptors(boundaries, rows, permutation_count)
    else:
        ends = np.cumsum(group_sizes); starts = np.r_[0, ends[:-1]]
        descriptors, boundaries, offset = [], [], 0
        for p, order in enumerate(permutations):
            groups = np.searchsorted(ends, order, side="right")
            group_order = groups[np.r_[True, groups[1:] != groups[:-1]]]
            expected_order = np.concatenate([np.arange(starts[g], ends[g]) for g in group_order])
            if len(group_order) != len(group_sizes) or not np.array_equal(order, expected_order):
                raise ValueError("Permutation must retain whole groups")
            if p < max(1, permutation_count - 1):
                folds = reference_group_folds([group_sizes[g] for g in group_order], fold_len_multiplier, min_fold_size)
                boundaries.append(folds)
                for prefix, end in folds:
                    descriptors.append([prefix, end, offset, p]); offset += end
        descriptors.append([rows, rows, offset, permutation_count - 1]); packed_rows = offset + rows
        descriptors = np.asarray(descriptors, np.uint32)
    rate, l2 = (float(np.float32(x)) for x in (learning_rate, l2_leaf_reg))
    if not math.isfinite(rate) or rate <= 0 or not math.isfinite(l2) or l2 < 0:
        raise ValueError("learning_rate must be positive and L2 nonnegative")
    # Upstream option normalization replaces zero L2 before leaf-step epsilon.
    l2 = l2 or float(np.float32(1e-20))
    base = (np.full(rows, float(np.float32(bias)), dtype=np.float64) if initial_predictions is None
            else np.asarray(initial_predictions, dtype=np.float32).astype(np.float64))
    if base.shape != labels.shape or not np.isfinite(base).all():
        raise ValueError("initial_predictions must have one finite value per row")
    cursors = np.empty(packed_rows, dtype=np.float64)
    for _, end, offset, permutation in descriptors:
        end, offset, permutation = map(int, (end, offset, permutation))
        cursors[offset:offset + end] = base[permutations[permutation, :end]]
    if initial_state is not None:
        previous = np.asarray(initial_state["cursors"], dtype=np.float64)
        if previous.shape != cursors.shape or not np.isfinite(previous).all():
            raise ValueError("initial_state cursors have an invalid shape or nonfinite values")
        for key, expected in (("folds", descriptors), ("permutations", permutations)):
            if key in initial_state and not np.array_equal(initial_state[key], expected):
                raise ValueError(f"initial_state {key} differs from this configuration")
        cursors[:] = previous
        if iteration_offset == 0:
            iteration_offset = int(initial_state.get("completed_iterations", 0))
    if not isinstance(iteration_offset, (int, np.integer)) or iteration_offset < 0:
        raise ValueError("iteration_offset must be a nonnegative integer")
    rng_state = None if initial_state is None else initial_state.get("selection_rng")
    if initial_state is not None and iteration_offset and rng_state is None:
        raise ValueError("resumed Ordered reference requires its prior variable-depth RNG state")
    selection_rng = OrderedSelectionRng(random_seed, permutation_count,
                                        initial_state=rng_state, iteration_offset=int(iteration_offset))
    result = {
        "depths": np.zeros(iterations, dtype=np.uint32),
        "split_features": np.zeros((iterations, depth), dtype=np.uint32),
        "split_bins": np.zeros((iterations, depth), dtype=np.uint32),
        "split_types": np.zeros((iterations, depth), dtype=np.uint8),
        "leaf_values": np.zeros((iterations, 1 << depth), dtype=np.float64),
        "leaf_weights": np.zeros((iterations, 1 << depth), dtype=np.float64),
        "task_leaf_values": np.zeros((iterations, len(descriptors), 1 << depth), dtype=np.float64),
        "selected_permutations": np.zeros(iterations, dtype=np.uint32),
        "loss": np.zeros(iterations + 1, dtype=np.float64),
        "backtracking_traces": [],
    }
    estimation_offset = int(descriptors[-1, 2])
    estimation_order = permutations[-1]

    def original_predictions():
        values = np.empty(rows, dtype=np.float64)
        values[estimation_order] = cursors[estimation_offset:estimation_offset + rows]
        return values

    result["loss"][0] = weighted_loss(labels, original_predictions(), weights, objective, objective_param)
    for tree in range(iterations):
        search = selection_rng.select()
        result["selected_permutations"][tree] = search
        search_folds = descriptors[:-1][descriptors[:-1, 3] == search]
        fold_derivatives = []
        for _, end, offset, permutation in search_folds:
            end, offset, permutation = map(int, (end, offset, permutation))
            order = permutations[permutation, :end]
            gradient, hessian = _derivatives(labels[order], cursors[offset:offset + end], weights[order], objective, objective_param)
            fold_derivatives.append((gradient, hessian if score_function == "NewtonCosine" else weights[order]))
        partitions = np.zeros((permutation_count, rows), dtype=np.int64)
        selected = set()
        score_before = 0.
        for level in range(depth):
            best_score, winner = float(np.finfo(np.float32).max), -1
            winner_score = best_score
            for candidate, (feature, border) in enumerate(zip(features, borders)):
                score = _score_candidate(banks[search], int(feature), int(border), partitions[search],
                    fold_derivatives, search_folds, permutations, l2,
                    fold_size_loss_normalization, 1 << level, bool(types[candidate]))
                score *= penalties[feature]
                gain = (score - score_before) * feature_weights[feature]
                if gain < best_score:
                    best_score, winner, winner_score = gain, candidate, score
            # Fully degenerate quality statistics produce a safe constant tree.
            if winner < 0:
                break
            feature, border = int(features[winner]), int(borders[winner])
            if (feature, border) in selected:
                break
            selected.add((feature, border))
            score_before = winner_score
            result["split_features"][tree, level] = feature
            result["split_bins"][tree, level] = border
            result["split_types"][tree, level] = types[winner]
            partitions |= (banks[:, feature] == border if types[winner] else banks[:, feature] > border).astype(np.int64) << level
            result["depths"][tree] += 1

        leaves = 1 << int(result["depths"][tree])
        coupled_values = None
        if leaf_estimation_backtracking != "No" and leaf_estimation_method != "Exact" and leaf_estimation_iterations > 1:
            task_inputs = []
            for prefix, _, offset, permutation in descriptors:
                prefix, offset, permutation = map(int, (prefix, offset, permutation))
                order = permutations[permutation, :prefix]
                task_inputs.append((labels[order], cursors[offset:offset + prefix], weights[order], partitions[permutation, order]))
            coupled_values, _, trace = estimate_tasks_reference(task_inputs, leaves,
                objective=objective, objective_param=objective_param,
                leaf_estimation_method=leaf_estimation_method, leaf_estimation_iterations=leaf_estimation_iterations,
                leaf_estimation_backtracking=leaf_estimation_backtracking, l2_leaf_reg=l2,
                fold_size_loss_normalization=fold_size_loss_normalization)
            result["backtracking_traces"].append(trace)
        else:
            result["backtracking_traces"].append([])
        for task, (prefix, end, offset, permutation) in enumerate(descriptors):
            prefix, end, offset, permutation = map(int, (prefix, end, offset, permutation))
            order = permutations[permutation, :end]
            estimate_order = order[:prefix]
            estimate_leaves = partitions[permutation, estimate_order]
            original_weights = weights[estimate_order]
            leaf_weights = np.bincount(estimate_leaves, weights=original_weights, minlength=leaves)
            values = np.zeros(leaves, dtype=np.float64)
            ridge = l2 * (original_weights.sum() if fold_size_loss_normalization else 1)
            if leaf_estimation_method == "Exact":
                for leaf in range(leaves):
                    members = estimate_leaves == leaf
                    values[leaf] = exact_leaf_value(labels[estimate_order][members],
                        cursors[offset:offset + prefix][members], original_weights[members],
                        objective, .5 if objective_param is None else objective_param)
            if coupled_values is not None:
                values = coupled_values[task].copy()
            for _ in range(0 if leaf_estimation_method == "Exact" or coupled_values is not None else leaf_estimation_iterations):
                gradient, hessian = _derivatives(labels[estimate_order],
                    cursors[offset:offset + prefix] + values[estimate_leaves], original_weights, objective, objective_param)
                sums = np.bincount(estimate_leaves, weights=gradient, minlength=leaves)
                denominator = (np.bincount(estimate_leaves, weights=hessian, minlength=leaves)
                               if leaf_estimation_method == "Newton" else leaf_weights)
                denominator = denominator + ridge
                values += np.divide(sums, denominator + 1e-20,
                                    out=np.zeros(leaves), where=denominator > 0)
                values[leaf_weights < 1e-20] = 0
            updates = rate * values
            result["task_leaf_values"][tree, task, :leaves] = updates
            cursors[offset:offset + end] += updates[partitions[permutation, order]]
            if task == len(descriptors) - 1:
                result["leaf_values"][tree, :leaves] = updates
                result["leaf_weights"][tree, :leaves] = leaf_weights
        result["loss"][tree + 1] = weighted_loss(labels, original_predictions(), weights, objective, objective_param)
        # A repeated/invalid winning split still consumed one attempted-depth
        # host seed. No candidates or requested depth zero consumes none.
        actual_depth = int(result["depths"][tree])
        attempts = min(depth, actual_depth + 1) if features.size and depth else 0
        selection_rng.finish(attempts)
    result.update(predictions=original_predictions(), cursors=cursors.copy(),
                  folds=descriptors, fold_boundaries=boundaries, permutations=permutations.copy(),
                  completed_iterations=int(iteration_offset) + iterations, selection_rng=selection_rng.state())
    result["rmse"] = result["loss"]
    result["state"] = {key: copy.deepcopy(result[key])
                       for key in ("cursors", "folds", "permutations", "completed_iterations", "selection_rng")}
    return result
