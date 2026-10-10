"""Weighted CUDA score-noise statistics and per-feature Metal random streams."""

import math

import numpy as np
import pytest

from test_bootstrap import probe, next_word, seed_for_item
from test_bootstrap_training import forbid_cpu_training, metal_device, _problem, _options, _assert_same_model
from catboost_metal import CatBoostMetalRegressor, _native


def _score_statistic(probe, gradients, weights):
    rows = len(gradients)
    result = probe(8, rows, derivatives=gradients, weights=weights)
    groups = min((rows + 255) // 256, 4096)
    return result["values"][:groups].sum(dtype=np.float64)


def test_zero_aware_division_and_row_count_normalization(probe):
    gradients = np.array([-2e-15, -1e-15, -.5e-15, 0, .5e-15, 1e-15, 2e-15, 1, 2, 3], np.float32)
    weights = np.array([1e-15, 1e-15, 1e-20, 0, 1e-20, 1e-15, 1e-15, 0, 2, 7], np.float32)
    weak = np.where(np.abs(gradients) < np.float32(1e-15), 0,
                    gradients / (weights + np.float32(1e-15)))
    expected = np.sum(weights.astype(np.float64) * weak.astype(np.float64)**2) / len(gradients)
    result = _score_statistic(probe, gradients, weights)
    assert result == pytest.approx(expected, rel=2e-6)
    # The two common mistaken formulas are observably different on this case.
    assert not math.isclose(result, np.mean(gradients.astype(np.float64)**2), rel_tol=.01)
    assert not math.isclose(result, expected * len(gradients) / weights.sum(), rel_tol=.01)


@pytest.mark.parametrize("rows", [1, 1031, 1000031])
def test_gpu_noise_statistic_matches_weighted_reference(probe, rows):
    rng = np.random.default_rng(9871)
    weights = rng.uniform(.1, 13, rows).astype(np.float32)
    targets = rng.normal(size=rows).astype(np.float32)
    gradients = (targets * weights).astype(np.float32)
    weights[::17] = 0
    gradients[::17] = 0
    expected = np.sum(weights.astype(np.float64) * targets.astype(np.float64)**2) / rows
    assert _score_statistic(probe, gradients, weights) == pytest.approx(expected, rel=2e-6, abs=1e-12)


def _normal(item, **kwargs):
    state = seed_for_item(item, **kwargs)
    for _ in range(4):
        state, _ = next_word(state)
    state, first = next_word(state)
    _, second = next_word(state)
    upper = np.nextafter(np.float32(1), np.float32(0))
    a = max(min(np.float32(first) * np.float32(2**-32), upper), np.float32(2**-32))
    b = min(np.float32(second) * np.float32(2**-32), upper)
    return math.sqrt(-2 * math.log(a)) * math.cos(float(np.float32(2 * math.pi)) * b)


@pytest.mark.parametrize("iteration,depth", [(0, 0), (0, 2), (53, 2)])
def test_feature_noise_has_separate_depth_and_iteration_streams(probe, iteration, depth):
    seed, features, scale = 0xa235b83412733291, 1031, .75
    expected = np.array([_normal(item, seed=seed, iteration=iteration, stream=depth + 1)
                         for item in range(features)]) * scale
    result = probe(9, features, seed=seed, iteration=iteration, stream=depth + 1, noise_scale=scale)["values"]
    np.testing.assert_allclose(result, expected, rtol=3e-5, atol=2e-6)
    repeated = probe(9, 17, seed=seed, iteration=iteration, stream=depth + 1, noise_scale=scale)["values"]
    np.testing.assert_array_equal(result[:17], repeated)


def test_zero_score_noise_scale_is_exact(probe):
    np.testing.assert_array_equal(probe(9, noise_scale=0)["values"], 0)


@pytest.mark.parametrize("iteration", [0, 8, 100000])
def test_noisy_split_winner_matches_weighted_feature_shared_cuda_formula(metal_device, iteration):
    rows, features, borders = 1031, 5, 7
    rng = np.random.default_rng(7128)
    bins = rng.integers(0, borders + 1, size=(features, rows), dtype=np.uint8)
    targets = rng.normal(size=rows).astype(np.float32)
    weights = rng.uniform(.25, 4, size=rows).astype(np.float32)
    weights[::19] = 0
    candidate_features = np.repeat(np.arange(features, dtype=np.uint32), borders)
    candidate_bins = np.tile(np.arange(borders, dtype=np.uint32), features)
    seed, strength, learning_rate, regularization = 612, 1.4, .2, 2.0
    result = _native.train(bins, targets, candidate_features, candidate_bins,
        iterations=1, depth=1, learning_rate=learning_rate, l2_leaf_reg=regularization,
        bias=0, score_function="Cosine", sample_weight=weights, random_strength=strength,
        random_seed=seed, iteration_offset=iteration)
    gradient = (weights * targets).astype(np.float32)
    weak = np.where(np.abs(gradient) < np.float32(1e-15), 0,
                    gradient / (weights + np.float32(1e-15)))
    stddev = math.sqrt(np.sum(weights.astype(np.float64) * weak.astype(np.float64)**2) / rows)
    exponent = iteration * learning_rate - math.log(rows)
    multiplier = 0 if exponent > 745 else 1 / (1 + math.exp(exponent))
    scale = np.float32(stddev * strength * multiplier)
    # One value per feature, reused across all its borders. The independent
    # scalar RNG also verifies the absolute-iteration decay and stream input.
    noise = np.array([np.float32(_normal(feature, seed=seed, iteration=iteration, stream=1)) * scale
                       for feature in range(features)], np.float32)
    scores = []
    for feature, border in zip(candidate_features, candidate_bins):
        left = bins[feature] <= border
        numerator, denominator = 0., 1e-10
        for selected in (left, ~left):
            weight = np.sum(weights[selected], dtype=np.float64)
            total = np.sum(gradient[selected], dtype=np.float64)
            mu = total / (weight + regularization) if weight > 0 else 0
            numerator += total * mu
            denominator += weight * mu * mu
        scores.append(np.float32(-numerator / math.sqrt(denominator)) + noise[feature])
    winner = int(np.argmin(scores))
    assert int(result.split_features[0, 0]) == int(candidate_features[winner])
    assert int(result.split_bins[0, 0]) == int(candidate_bins[winner])


def test_score_noise_changes_cosine_selection_and_l2_ignores_it(metal_device):
    features, target, weights = _problem()
    cosine = CatBoostMetalRegressor(**_options(score_function="Cosine", random_strength=0)).fit(
        features, target, sample_weight=weights)
    noisy = CatBoostMetalRegressor(**_options(score_function="Cosine", random_strength=100)).fit(
        features, target, sample_weight=weights)
    assert np.max(np.abs(cosine.training_predictions_ - noisy.training_predictions_)) > 1e-5
    l2 = CatBoostMetalRegressor(**_options(score_function="L2", random_strength=0)).fit(
        features, target, sample_weight=weights)
    l2_noisy = CatBoostMetalRegressor(**_options(score_function="L2", random_strength=100)).fit(
        features, target, sample_weight=weights)
    _assert_same_model(l2, l2_noisy)


@pytest.mark.parametrize("bootstrap", [{"bootstrap_type": "No"}, {"bootstrap_type": "MVS", "subsample": .6}])
def test_randomized_score_snapshot_resume_matches_uninterrupted_training(metal_device, tmp_path, bootstrap):
    features, target, weights = _problem()
    options = _options(iterations=7, score_function="Cosine", random_strength=2, **bootstrap)
    path = tmp_path / "score-noise.snapshot"
    CatBoostMetalRegressor(**(options | {"iterations": 3})).fit(
        features, target, sample_weight=weights, save_snapshot=True, snapshot_file=path)
    resumed = CatBoostMetalRegressor(**options).fit(features, target, sample_weight=weights,
        save_snapshot=True, snapshot_file=path)
    direct = CatBoostMetalRegressor(**options).fit(features, target, sample_weight=weights)
    _assert_same_model(resumed, direct)
    assert resumed.training_stats_["resumed_iterations"] == 3
