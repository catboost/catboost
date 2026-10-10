"""Scalar arithmetic for CUDA Plain SolarL2, SatL2 and leave-one-out L2 scores.

Sources: cuda/methods/kernel/score_calcers.cuh::{TSolarScoreCalcer,
TLOOL2ScoreCalcer,TSatL2ScoreCalcer} and pointwise_scores.cu::FindOptimalSplitSingleFoldImpl.
All use original sample weights, ignore the constructor's L2 argument, and
omit random score noise. They accumulate child contributions in float32;
CUDA AddLeaf takes double inputs and LOOL2 explicitly rounds its adjustment
to float32 before squaring it. Reductions of observations here use float64.

These are Plain single-fold formulas. Ordered Solar uses a separate
learn/test-fold formula in FindOptimalSplitSolarImpl; it is not this score.
No training backend is imported.
"""

import math

import numpy as np


def score_children(sums, weights, score_function):
    """Return the minimized score for interleaved left/right leaf statistics."""
    if score_function not in ("SolarL2", "LOOL2", "SatL2"):
        raise ValueError("auxiliary score must be SolarL2, LOOL2 or SatL2")
    score = np.float32(0)
    for total, weight in zip(sums, weights):
        total, weight = float(total), float(weight)
        contribution = 0.0
        if score_function == "SolarL2":
            if weight > float(np.float32(1e-20)):
                contribution = -total * total * (1 + 2 * math.log1p(weight)) / weight
        elif score_function == "SatL2":
            if weight > 2:
                adjustment = np.float32(weight * (weight - 2) / (weight * weight - 3 * weight + 1))
                contribution = float(adjustment) * (-total * total / weight)
        elif weight > 1:
            adjustment = np.float32(weight / (weight - 1))
            adjustment = np.float32(adjustment * adjustment)
            contribution = -total * total * float(adjustment) / weight
        score = np.float32(float(score) + contribution)
    return float(score)


def candidate_scores(bins, gradients, weights, features, borders, score_function):
    """Score each root split independently, retaining input candidate order."""
    gradients, weights = np.asarray(gradients, np.float64), np.asarray(weights, np.float64)
    result = []
    for feature, border in zip(features, borders):
        left = bins[feature] <= border
        totals = [math.fsum(gradients[left]), math.fsum(gradients[~left])]
        masses = [math.fsum(weights[left]), math.fsum(weights[~left])]
        result.append(score_children(totals, masses, score_function))
    return np.asarray(result, np.float32)
