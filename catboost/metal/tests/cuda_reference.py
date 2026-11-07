"""Small, independent oracle for the numeric CUDA-to-Metal translation.

This is a scalar interpretation of these upstream CUDA sources, not a call to
the CatBoost CPU trainer:

* ``cuda/methods/kernel/score_calcers.cuh``: ``TL2ScoreCalcer`` and
  ``TCosineScoreCalcer`` (normalization disabled, random strength zero).
* ``cuda/methods/kernel/pointwise_scores.cu``:
  ``FindOptimalSplitSingleFoldImpl``; score every candidate, add both children
  of every current leaf, minimize score, and break ties by candidate index.
* ``cuda/methods/kernel/split_properties_helpers.cuh``:
  ``ScanHistogramsImpl``; a candidate's left child includes bins <= its border.
* ``cuda/methods/oblivious_tree_doc_parallel_structure_searcher.cpp``:
  ``FitImpl``; stop BEFORE adding a winning split already in the tree.
* ``cuda/targets/kernel/pointwise_targets.cu``: ``TRmseTarget``; the weak target is
  target minus prediction, with Hessian one for unweighted RMSE.
* ``cuda/methods/leaves_estimation/oblivious_tree_leaves_estimator.cpp``:
  ``WriteSecondDerivatives``, together with ``descent_helpers.cpp``; one
  Newton leaf update is gradient sum / (weight + L2). Both the split and leaf
  estimators use L2 unchanged, with their optional normalization disabled.

Only unweighted RMSE, Plain boosting, symmetric trees, no sampling, and one
Newton update per tree are represented. Float64 reductions make this an
arithmetic reference; float32 Metal results require numerical tolerances.
There is no positive-gain condition or nonempty-child filter. With no split
candidates we deliberately emit depth-zero trees, the Metal API's extension
for constant-feature data. The upstream searcher instead requires a candidate.
"""

from __future__ import annotations

import math

import numpy as np


def _score_children(sums: np.ndarray, weights: np.ndarray, l2: float,
                    score_function: str) -> float:
    """Translate the two score calculators; smaller is better.

    ``sums``/``weights`` are interleaved [left0, right0, left1, right1, ...].
    L2's upstream default meta exponent is one. Cosine's denominator starts
    at 1e-10, as in TCosineScoreCalcer::NextFeature, rather than zero.
    """
    if score_function == "L2":
        score = 0.0
        for total, weight in zip(sums, weights):
            if weight > 1e-20:
                score -= float(total) * float(total) / (float(weight) + l2)
        return score

    numerator = 0.0
    denominator_squared = 1e-10
    for total, weight in zip(sums, weights):
        total = float(total)
        weight = float(weight)
        mean = total / (weight + l2) if weight > 0.0 else 0.0
        numerator += total * mean
        denominator_squared += weight * mean * mean
    return -numerator / math.sqrt(denominator_squared)


def train_reference(
    bins: np.ndarray,
    targets: np.ndarray,
    candidate_features: np.ndarray,
    candidate_bins: np.ndarray,
    *,
    iterations: int,
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    bias: float,
    score_function: str = "Cosine",
) -> dict[str, np.ndarray]:
    """Train the restricted CUDA arithmetic reference on quantized input.

    Input bins have feature-major shape [features, rows]. Candidate arrays
    contain parallel feature and border indices in the exact order used by
    the Metal backend. First split is leaf-index bit zero. Tree arrays have
    fixed strides: unused splits and leaves are zero-filled. Leaf values
    include learning_rate; predictions include bias. Floating output arrays
    use float64, while depth/split arrays use uint32.

    In FindOptimalSplitSingleFoldImpl, gain subtracts scoreBeforeSplit and
    applies feature penalties. With all penalties one, subtraction is common
    to every candidate, so directly minimizing the score gives the same
    mathematical ordering.
    """
    bins = np.asarray(bins)
    targets = np.asarray(targets)
    candidate_features = np.asarray(candidate_features)
    candidate_bins = np.asarray(candidate_bins)
    if bins.ndim != 2 or bins.dtype != np.uint8:
        raise ValueError("bins must be a uint8 array with shape [features, rows]")
    feature_count, row_count = bins.shape
    if row_count == 0 or targets.shape != (row_count,):
        raise ValueError("targets must contain one value per nonempty input row")
    if not np.all(np.isfinite(targets)):
        raise ValueError("targets must be finite")
    if (candidate_features.ndim != 1 or candidate_bins.ndim != 1
            or candidate_features.shape != candidate_bins.shape):
        raise ValueError("candidate arrays must have matching one-dimensional shapes")
    if (candidate_features.dtype != np.uint32
            or candidate_bins.dtype != np.uint32):
        raise ValueError("candidate arrays must use uint32")
    if (np.any(candidate_features >= feature_count)
            or np.any(candidate_bins > 255)):
        raise ValueError("candidate feature or bin is out of range")
    if (not isinstance(iterations, (int, np.integer)) or iterations < 0
            or not isinstance(depth, (int, np.integer)) or not 0 <= depth <= 16):
        raise ValueError("iterations must be nonnegative and depth must be in [0, 16]")
    if score_function not in ("L2", "Cosine"):
        raise ValueError("score_function must be L2 or Cosine")
    if (not math.isfinite(learning_rate) or learning_rate <= 0
            or not math.isfinite(l2_leaf_reg) or l2_leaf_reg < 0
            or not math.isfinite(bias)):
        raise ValueError("learning_rate, l2_leaf_reg, or bias is invalid")

    # ABI scalar arguments and labels arrive at the device as float32. Honor
    # that representation before doing our independent double reductions.
    rate = float(np.float32(learning_rate))
    regularization = float(np.float32(l2_leaf_reg))
    labels = targets.astype(np.float32).astype(np.float64)
    predictions = np.full(row_count, float(np.float32(bias)), dtype=np.float64)
    max_leaves = 1 << depth
    tree_depths = np.zeros(iterations, dtype=np.uint32)
    split_features = np.zeros((iterations, depth), dtype=np.uint32)
    split_bins = np.zeros((iterations, depth), dtype=np.uint32)
    leaf_values = np.zeros((iterations, max_leaves), dtype=np.float64)
    leaf_weights = np.zeros((iterations, max_leaves), dtype=np.float64)
    rmse = np.empty(iterations + 1, dtype=np.float64)
    rmse[0] = math.sqrt(float(np.mean((labels - predictions) ** 2)))

    for tree in range(iterations):
        gradients = labels - predictions
        partitions = np.zeros(row_count, dtype=np.int64)
        selected: set[tuple[int, int]] = set()
        actual_depth = 0

        for level in range(depth):
            if candidate_features.size == 0:
                break
            current_leaves = 1 << level
            parent_weights = np.bincount(partitions, minlength=current_leaves)
            parent_sums = np.bincount(
                partitions, weights=gradients, minlength=current_leaves
            )
            best_score = math.inf
            best_candidate = -1

            # Independent masked reductions avoid reproducing the histogram
            # kernel implementation in its own oracle.
            for candidate, (feature, border) in enumerate(
                    zip(candidate_features, candidate_bins)):
                left = bins[int(feature)] <= int(border)
                left_weights = np.bincount(
                    partitions[left], minlength=current_leaves
                )
                left_sums = np.bincount(
                    partitions[left], weights=gradients[left], minlength=current_leaves
                )
                child_weights = np.empty(2 * current_leaves, dtype=np.float64)
                child_sums = np.empty(2 * current_leaves, dtype=np.float64)
                child_weights[0::2] = left_weights
                child_weights[1::2] = np.maximum(parent_weights - left_weights, 0)
                child_sums[0::2] = left_sums
                child_sums[1::2] = parent_sums - left_sums
                score = _score_children(
                    child_sums, child_weights, regularization, score_function
                )
                # Strict comparison retains the earliest candidate on ties,
                # matching CUDA's explicit candidate-index reduction tie break.
                if score < best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate < 0:
                raise ArithmeticError("all candidate scores are nonfinite")
            feature = int(candidate_features[best_candidate])
            border = int(candidate_bins[best_candidate])
            if (feature, border) in selected:
                break
            selected.add((feature, border))
            split_features[tree, level] = feature
            split_bins[tree, level] = border
            partitions |= (bins[feature] > border).astype(np.int64) << level
            actual_depth += 1

        tree_depths[tree] = actual_depth
        leaf_count = 1 << actual_depth
        counts = np.bincount(partitions, minlength=leaf_count)
        sums = np.bincount(partitions, weights=gradients, minlength=leaf_count)
        updates = np.zeros(leaf_count, dtype=np.float64)
        occupied = counts > 0
        updates[occupied] = rate * sums[occupied] / (counts[occupied] + regularization)
        leaf_values[tree, :leaf_count] = updates
        leaf_weights[tree, :leaf_count] = counts
        predictions += updates[partitions]
        rmse[tree + 1] = math.sqrt(float(np.mean((labels - predictions) ** 2)))

    return {
        "depths": tree_depths,
        "split_features": split_features,
        "split_bins": split_bins,
        "leaf_values": leaf_values,
        "leaf_weights": leaf_weights,
        "predictions": predictions,
        "rmse": rmse,
    }
