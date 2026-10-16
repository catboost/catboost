"""Independent scalar oracle for weighted CUDA RMSE and binary objectives.

Sources relative to catboost/:

* cuda/targets/kernel/pointwise_targets.cu: TRmseTarget, CrossEntropyImpl.
* cuda/targets/pointwise_target_impl.h: GradientAt retains sample weights for
  L2/Cosine structure scoring; NewtonAt uses objective Hessians instead.
* cuda/methods/kernel/score_calcers.cuh: TL2ScoreCalcer, TCosineScoreCalcer.
* cuda/methods/kernel/pointwise_scores.cu: FindOptimalSplitSingleFoldImpl.
* cuda/methods/oblivious_tree_doc_parallel_structure_searcher.cpp: FitImpl
  stops before adding a winning split that has already appeared.
* cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp:
  WriteSecondDerivatives adds lambda to each Hessian; optional derivative
  normalization and AddRidgeToTargetFunction are disabled by default.
* cuda/methods/leaves_estimation/descent_helpers.cpp: TNewtonLikeWalker and
  TDirectionEstimator, with step_estimator.cpp's TSkipStepEstimation (No).
* cuda/methods/doc_parallel_boosting.h: rescale the final tree once by the
  learning rate, after all unshrunk Newton leaf steps.

This module does not import the native backend or train a CPU CatBoost model.
Reductions use float64 deliberately; binary probabilities retain CUDA's float32
saturation boundary. It is not a bit-identical CUDA reduction simulation.
Stable loss evaluation intentionally avoids CUDA's extreme-logit cancellation.
"""

import math

import numpy as np

from cuda_reference import _score_children


def sigmoid(raw):
    """Stable scalar-equivalent logistic transform for extreme finite inputs."""
    raw = np.asarray(raw, dtype=np.float64)
    probability = np.empty_like(raw)
    positive = raw >= 0
    probability[positive] = 1 / (1 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    probability[~positive] = exponential / (1 + exponential)
    return probability


def weighted_loss(targets, raw_predictions, weights, objective):
    """Weighted RMSE or binary log loss, divided by SUM OF WEIGHTS."""
    targets = np.asarray(targets, dtype=np.float64)
    raw_predictions = np.asarray(raw_predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if objective == "RMSE":
        return math.sqrt(float(np.dot(weights, (targets - raw_predictions) ** 2)
                               / weights.sum()))
    if objective == "Logloss":
        targets = (targets > 0.5).astype(np.float64)
    terms = (targets * np.logaddexp(0, -raw_predictions)
             + (1 - targets) * np.logaddexp(0, raw_predictions))
    return float(np.dot(weights, terms) / weights.sum())


def initial_bias(targets, weights=None, *, objective="RMSE", boost_from_average=True):
    """Host initialization from optimal_const_for_loss.h; labels already encoded.

    CUDA multiplies float targets and weights before its double accumulation.
    There is no implicit clipping of binary means.
    """
    if not boost_from_average:
        return 0.0
    targets = np.asarray(targets, dtype=np.float32).astype(np.float64)
    weights = (np.ones(targets.size, dtype=np.float64) if weights is None
               else np.asarray(weights, dtype=np.float32).astype(np.float64))
    products = (weights.astype(np.float32) * targets.astype(np.float32)).astype(np.float64)
    mean = float(np.float32(products.sum() / weights.sum()))
    if objective == "RMSE":
        return mean
    if not 0 < mean < 1:
        raise ValueError("binary boost_from_average requires a mean strictly between 0 and 1")
    return float(np.float32(math.log(mean / (1 - mean))))


def _derivatives(labels, raw, weights, objective):
    if objective == "RMSE":
        return weights * (labels - raw), weights.copy()
    probability = np.maximum(sigmoid(raw).astype(np.float32), np.float32(1e-40)).astype(np.float64)
    return weights * (labels - probability), weights * probability * (1 - probability)


def train_reference(
    bins, targets, candidate_features, candidate_bins, *, iterations, depth,
    learning_rate, l2_leaf_reg, bias, score_function="Cosine", objective="RMSE",
    sample_weight=None, leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
):
    """Translate the restricted weighted training semantics independently.

    Input bins are uint8 [features, rows]. Candidate ordering breaks score ties.
    Leaf values include shrinkage; predictions remain raw margins for binary
    objectives. ``loss`` contains RMSE for RMSE and log loss for either binary
    objective. ``rmse`` aliases loss only for compatibility with the first ABI.
    """
    if objective not in ("RMSE", "Logloss", "CrossEntropy"):
        raise ValueError("unsupported oracle objective")
    if score_function not in ("L2", "Cosine"):
        raise ValueError("unsupported oracle score function")
    if leaf_estimation_backtracking != "No":
        raise ValueError("this oracle only implements explicitly disabled backtracking")
    if leaf_estimation_iterations < 1:
        raise ValueError("leaf_estimation_iterations must be positive")
    bins = np.asarray(bins, dtype=np.uint8)
    labels = np.asarray(targets, dtype=np.float32).astype(np.float64)
    if objective == "Logloss":
        labels = (labels > 0.5).astype(np.float64)
    features = np.asarray(candidate_features, dtype=np.uint32)
    borders = np.asarray(candidate_bins, dtype=np.uint32)
    rows = labels.size
    weights = (np.ones(rows, dtype=np.float64) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float32).astype(np.float64))
    if weights.shape != labels.shape or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("weights must match targets, be nonnegative, and have positive sum")
    rate = float(np.float32(learning_rate))
    regularization = float(np.float32(l2_leaf_reg))
    # private/libs/options/catboost_options.cpp replaces an explicitly zero L2
    # with 1e-20f before the walker's separate denominator epsilon is added.
    if regularization == 0:
        regularization = float(np.float32(1e-20))
    predictions = np.full(rows, float(np.float32(bias)), dtype=np.float64)
    result = {
        "depths": np.zeros(iterations, dtype=np.uint32),
        "split_features": np.zeros((iterations, depth), dtype=np.uint32),
        "split_bins": np.zeros((iterations, depth), dtype=np.uint32),
        "leaf_values": np.zeros((iterations, 1 << depth), dtype=np.float64),
        "leaf_weights": np.zeros((iterations, 1 << depth), dtype=np.float64),
        "loss": np.zeros(iterations + 1, dtype=np.float64),
    }
    result["loss"][0] = weighted_loss(labels, predictions, weights, objective)

    for tree in range(iterations):
        gradients, _ = _derivatives(labels, predictions, weights, objective)
        partitions = np.zeros(rows, dtype=np.int64)
        selected = set()
        actual_depth = 0
        for level in range(depth):
            if features.size == 0:
                break
            count = 1 << level
            parent_weights = np.bincount(partitions, weights=weights, minlength=count)
            parent_sums = np.bincount(partitions, weights=gradients, minlength=count)
            best_score, winner = math.inf, -1
            for candidate, (feature, border) in enumerate(zip(features, borders)):
                left = bins[int(feature)] <= int(border)
                left_weights = np.bincount(partitions[left], weights=weights[left], minlength=count)
                left_sums = np.bincount(partitions[left], weights=gradients[left], minlength=count)
                child_weights = np.empty(2 * count, dtype=np.float64)
                child_sums = np.empty(2 * count, dtype=np.float64)
                child_weights[0::2] = left_weights
                child_weights[1::2] = np.maximum(parent_weights - left_weights, 0)
                child_sums[0::2] = left_sums
                child_sums[1::2] = parent_sums - left_sums
                score = _score_children(child_sums, child_weights, regularization, score_function)
                if score < best_score:
                    best_score, winner = score, candidate
            if winner < 0:
                raise ArithmeticError("all candidate scores are nonfinite")
            feature, border = int(features[winner]), int(borders[winner])
            if (feature, border) in selected:
                break
            selected.add((feature, border))
            result["split_features"][tree, level] = feature
            result["split_bins"][tree, level] = border
            partitions |= (bins[feature] > border).astype(np.int64) << level
            actual_depth += 1

        count = 1 << actual_depth
        leaf_weights = np.bincount(partitions, weights=weights, minlength=count)
        unshrunk = np.zeros(count, dtype=np.float64)
        for _ in range(leaf_estimation_iterations):
            gradients, hessians = _derivatives(
                labels, predictions + unshrunk[partitions], weights, objective)
            sums = np.bincount(partitions, weights=gradients, minlength=count)
            hessian_sums = np.bincount(partitions, weights=hessians, minlength=count)
            regularized = hessian_sums + regularization
            direction = np.divide(sums, regularized + 1e-20,
                                  out=np.zeros(count), where=regularized > 0)
            unshrunk += direction
            unshrunk[leaf_weights < 1e-20] = 0
        updates = rate * unshrunk
        result["depths"][tree] = actual_depth
        result["leaf_values"][tree, :count] = updates
        result["leaf_weights"][tree, :count] = leaf_weights
        predictions += updates[partitions]
        result["loss"][tree + 1] = weighted_loss(labels, predictions, weights, objective)

    result["predictions"] = predictions
    result["rmse"] = result["loss"]
    return result
