"""Seeded greedy structure sampling, original-weight leaves, and CUDA terminals."""
import math
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal import _greedy
from test_bootstrap import uniforms, seed_for_item, next_word
from test_greedy_training import data, derivatives, expected_values, route


@pytest.fixture(autouse=True)
def metal_without_cpu_fit(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs): raise AssertionError("CPU fitting is forbidden")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier): monkeypatch.setattr(cls, "fit", forbidden)


def draws(kind, rows, seed=857, iteration=0, subsample=.43, temperature=1.3):
    if kind == "No": return np.ones(rows, np.float32)
    if kind == "Bayesian":
        return ((-np.log(uniforms(rows, seed=seed, iteration=iteration).astype(float) + 1e-20))**temperature).astype(np.float32)
    if kind == "Bernoulli": return (uniforms(rows, seed=seed, iteration=iteration) < np.float32(subsample)).astype(np.float32)
    rate = np.float32(-math.log(1 - np.float32(subsample)))
    values = []
    for row in range(rows):
        state = seed_for_item(row, seed=seed, iteration=iteration)
        total, count = np.float32(0), 0
        while True:
            state, word = next_word(state)
            uniform = min(np.float32(word) * np.float32(2**-32), np.nextafter(np.float32(1), np.float32(0)))
            total += np.float32(math.log(max(uniform, np.float32(2**-32)))); count += 1
            if total <= -rate: break
        values.append(count - 1)
    return np.array(values, np.float32)


def quality(gradient, weight, score, l2):
    if score == "SolarL2":
        return -sum(g*g / w * (1 + 2 * np.log1p(w)) if w > 1e-20 else 0 for g, w in zip(gradient, weight))
    if score == "LOOL2":
        return -sum(g*g / w * (w/(w-1))**2 if w > 1 else 0 for g, w in zip(gradient, weight))
    if score == "SatL2":
        return -sum(g*g / w * w*(w-2)/(w*w-3*w+1) if w > 2 else 0 for g, w in zip(gradient, weight))
    if score.endswith("L2"):
        return -sum(g*g / (w+l2) if w > 1e-20 else 0 for g, w in zip(gradient, weight))
    mu = [g/(w+l2) if w > 0 else 0 for g, w in zip(gradient, weight)]
    return -np.dot(gradient, mu) / np.sqrt(1e-10 + np.dot(weight, np.square(mu)))


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2", "SatL2"])
@pytest.mark.parametrize("kind", ["No", "Bayesian", "Bernoulli", "Poisson"])
@pytest.mark.parametrize("objective", ["RMSE", "Logloss"])
def test_root_split_uses_seeded_sampled_gradient_and_correct_denominator(score, kind, objective):
    bins, y, weights, cf, cb = data(objective)
    multipliers = draws(kind, len(y), iteration=17)
    gradient, hessian = derivatives(np.full(len(y), .1), y.astype(float), weights.astype(float), objective)
    gradient = (gradient.astype(np.float32) * multipliers).astype(np.float32).astype(float)
    denominator = ((hessian.astype(np.float32) if score.startswith("Newton") else weights) * multipliers).astype(np.float32).astype(float)
    before = quality([gradient.sum()], [denominator.sum()], score, 2.3)
    scores = []
    for feature, border in zip(cf, cb):
        left = bins[feature] <= border
        gs, ws = [gradient[left].sum(), gradient[~left].sum()], [denominator[left].sum(), denominator[~left].sum()]
        scores.append(quality(gs, ws, score, 2.3) - before if min(ws) >= 1e-20 else 0.)
    winner = int(np.argmin(scores))
    result = _greedy.train(bins, y, cf, cb, objective=objective, sample_weight=weights,
        iterations=1, depth=1, max_leaves=2, grow_policy="Lossguide", score_function=score,
        bias=.1, l2_leaf_reg=2.3, bootstrap_type=kind, random_seed=857, iteration_offset=17,
        subsample=.43, bagging_temperature=1.3)
    assert result.trees[0].nodes[0, :3].tolist() == [int(cf[winner]), int(cb[winner]), 0]


@pytest.mark.parametrize("kind", ["Bayesian", "Bernoulli", "Poisson"])
@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_sampled_structure_keeps_original_final_leaf_weights(kind, policy, method):
    bins, y, weights, cf, cb = data("Logloss")
    result = _greedy.train(bins, y, cf, cb, objective="Logloss", sample_weight=weights,
        iterations=2, depth=4, max_leaves=11, grow_policy=policy, score_function="NewtonCosine",
        bias=.1, l2_leaf_reg=2.3, learning_rate=.2, bootstrap_type=kind, random_seed=857,
        subsample=.43, bagging_temperature=1.3, leaf_estimation_method=method, leaf_estimation_iterations=3)
    prediction = np.full(len(y), .1)
    for tree in result.trees:
        values, expected_weights, ids = expected_values(tree, bins, y.astype(float), weights.astype(float),
            prediction, objective="Logloss", method=method, leaf_iterations=3, l2=2.3, rate=.2)
        np.testing.assert_allclose(tree.leaf_values, values, rtol=4e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_weights, expected_weights, rtol=3e-6, atol=2e-5)
        prediction += tree.leaf_values[ids]
    np.testing.assert_allclose(result.predictions, prediction, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("kind", ["Bernoulli", "Poisson"])
def test_min_data_counts_sampled_rows_even_when_original_weights_are_zero(kind):
    rows = 128
    kept = int(np.count_nonzero(draws(kind, rows, subsample=.31)))
    assert 2 < kept < rows
    weights = np.zeros(rows, np.float32); weights[::17] = 1
    options = dict(iterations=1, depth=5, max_leaves=16, grow_policy="Lossguide", score_function="L2",
        sample_weight=weights, bootstrap_type=kind, random_seed=857, subsample=.31)
    bins = np.zeros((1, rows), np.uint8)
    stopped = _greedy.train(bins, np.ones(rows), [0], [0], min_data_in_leaf=kept, **options)
    growing = _greedy.train(bins, np.ones(rows), [0], [0], min_data_in_leaf=kept-1, **options)
    assert len(stopped.trees[0].leaf_values) == 2
    assert len(growing.trees[0].leaf_values) == 6


@pytest.mark.parametrize("kind,options", [("Bayesian", {"bagging_temperature": 0.}),
                                        ("Bernoulli", {"subsample": 1.})])
def test_identity_sampling_and_l2_noise_are_noops(kind, options):
    bins, y, weights, cf, cb = data()
    standard = dict(iterations=3, depth=3, max_leaves=7, sample_weight=weights, score_function="SatL2")
    baseline = _greedy.train(bins, y, cf, cb, **standard)
    sampled = _greedy.train(bins, y, cf, cb, **standard, bootstrap_type=kind, random_strength=100., **options)
    np.testing.assert_array_equal(sampled.predictions, baseline.predictions)
    for left, right in zip(sampled.trees, baseline.trees): np.testing.assert_array_equal(left.nodes, right.nodes)


def test_fully_filtered_draw_remains_finite_and_estimates_on_original_rows():
    assert not np.any(draws("Bernoulli", 2, subsample=1e-6))
    result = _greedy.train(np.zeros((1, 2), np.uint8), [1., 2.], [0], [0],
        iterations=1, depth=3, bootstrap_type="Bernoulli", random_seed=857, subsample=1e-6)
    assert np.isfinite(result.predictions).all()
    assert result.trees[0].leaf_weights.sum() == 2
    assert len(result.trees[0].leaf_values) == 2
