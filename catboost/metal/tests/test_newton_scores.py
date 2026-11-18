"""Newton DocParallel split scores, bootstrap denominators and score noise.

Source: oblivious_tree_doc_parallel_structure_searcher.cpp::ComputeWeakTarget
chooses weighted Hessians before ComputeScoreStdDev and BootstrapAndFilter.
pointwise_scores.cu aliases NewtonL2/L2 and NewtonCosine/Cosine calculators.
Leaf estimation independently consumes original targets and sample weights.
"""

import math
import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor, _native
from cuda_reference import _score_children
from cuda_scalar_reference import objective_terms, train_reference
from test_bootstrap import uniforms, _mvs_threshold
from test_score_noise import _normal


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Newton score checks cannot invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Newton score checks require Apple Silicon")
    return _native.device_info()


def options(objective="Logloss", score="NewtonCosine", **overrides):
    result = dict(iterations=1, depth=1, learning_rate=0.2, l2_leaf_reg=3, bias=0,
                  score_function=score, objective=objective, objective_param=None,
                  leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No")
    result.update(overrides)
    return result


def compare(bins, targets, features, borders, **kwargs):
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for tree, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :depth], expected["split_features"][tree, :depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :depth], expected["split_bins"][tree, :depth])
    for key in ("leaf_weights", "leaf_values", "predictions"):
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=1e-4, atol=1e-4, err_msg=key)
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=1e-4, atol=1e-5)
    return actual


def _statistics(targets, weights, objective, param, bias, score):
    _, gradient, hessian = objective_terms(targets, np.full(len(targets), np.float32(bias)), objective, param)
    gradient = (weights * gradient).astype(np.float32)
    denominator = (weights * hessian).astype(np.float32) if score.startswith("Newton") else weights.astype(np.float32)
    return gradient, denominator


def _scores(bins, gradients, denominators, features, borders, regularization, score, noise=None):
    scores = []
    for feature, border in zip(features, borders):
        left = bins[feature] <= border
        sums = np.array([gradients[left].sum(dtype=np.float64), gradients[~left].sum(dtype=np.float64)])
        weights = np.array([denominators[left].sum(dtype=np.float64), denominators[~left].sum(dtype=np.float64)])
        value = np.float32(_score_children(sums, weights, regularization, score.removeprefix("Newton")))
        if noise is not None and score.endswith("Cosine"):
            value += noise[feature]
        scores.append(value)
    return np.array(scores)


WINNER_FIXTURES = [
    ("Logloss", [1, 1, 0, 1, 1, 1, 0, 1], [1, 1, 3, 3, 1, 1, 1, 3], -math.log(4), 1, 0),
    ("Poisson", [3, 7, 4, 0, 0, 2, 6, 2], [1, 1, 3, 2, 1, 1, 3, 3], math.log(4), 0, 1),
]


@pytest.mark.parametrize("family", ["L2", "Cosine"])
@pytest.mark.parametrize("objective,targets,weights,bias,first_order_winner,newton_winner", WINNER_FIXTURES)
def test_newton_curvature_changes_the_split_winner(metal_device, family, objective, targets, weights, bias,
                                                  first_order_winner, newton_winner):
    bins = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 0, 0, 1, 1]], np.uint8)
    y, weights = np.array(targets, np.float32), np.array(weights, np.float32)
    features, borders = np.array([0, 1], np.uint32), np.zeros(2, np.uint32)
    first = compare(bins, y, features, borders, **options(objective, family, bias=bias, sample_weight=weights))
    newton = compare(bins, y, features, borders, **options(objective, "Newton" + family, bias=bias, sample_weight=weights))
    assert first.split_features[0, 0] == first_order_winner
    assert newton.split_features[0, 0] == newton_winner
    assert first_order_winner != newton_winner


@pytest.mark.parametrize("objective,param", [("RMSE", None), ("Logloss", None), ("Poisson", None),
                                           ("Huber", 0.8), ("Tweedie", 1.5)])
@pytest.mark.parametrize("score", ["NewtonL2", "NewtonCosine"])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_weighted_multitree_newton_scores_match_reference(metal_device, objective, param, score, method):
    rng = np.random.default_rng(774)
    bins = rng.integers(0, 8, (4, 263), dtype=np.uint8)
    signal = 0.8 * (bins[0] > 3) - 0.5 * (bins[2] > 4) + rng.normal(0, 0.1, 263)
    y = ((signal > 0.2).astype(float) if objective == "Logloss" else
         np.exp(signal) if objective in ("Poisson", "Tweedie") else signal).astype(np.float32)
    weights = (0.25 + rng.random(263) * 2).astype(np.float32)
    weights[::17] = 0
    compare(bins, y, np.repeat(np.arange(4, dtype=np.uint32), 7),
             np.tile(np.arange(7, dtype=np.uint32), 4),
             **options(objective, score, iterations=3, depth=3, objective_param=param,
                       sample_weight=weights, leaf_estimation_method=method, leaf_estimation_iterations=3))


def _bootstrap_factors(kind, gradients, seed, fraction=0.6, regularization=0.75):
    uniform = uniforms(len(gradients), seed=seed)
    if kind == "No":
        return np.ones(len(gradients), np.float32)
    if kind == "Bernoulli":
        return (uniform < np.float32(fraction)).astype(np.float32)
    if kind == "Bayesian":
        return -np.log(uniform + np.float32(1e-20))
    magnitudes = np.sqrt(gradients.astype(np.float64) ** 2 + regularization)
    threshold = _mvs_threshold(magnitudes, float(np.float32(fraction)))
    probability = np.minimum(1, magnitudes / threshold)
    included = (probability > np.finfo(np.float32).eps) & (uniform < probability)
    return np.divide(1, probability, out=np.zeros(len(gradients)), where=included).astype(np.float32)


@pytest.mark.parametrize("kind", ["No", "Bernoulli", "Bayesian", "MVS"])
@pytest.mark.parametrize("score", ["NewtonL2", "NewtonCosine"])
def test_bootstrap_and_noise_use_hessian_structure_weights_but_original_leaf_weights(metal_device, kind, score):
    rng = np.random.default_rng(148)
    bins = rng.integers(0, 8, (5, 1031), dtype=np.uint8)
    targets = (rng.random(1031) < 0.4).astype(np.float32)
    weights = rng.uniform(0.1, 3, 1031).astype(np.float32)
    weights[::19] = 0
    seed, bias, strength, regularization = 0, -1.1, (0.3 if kind == "No" else 1.3), 3
    features = np.repeat(np.arange(5, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 5)
    gradient, denominator = _statistics(targets, weights, "Logloss", None, bias, score)
    factors = _bootstrap_factors(kind, gradient, seed)
    weak = np.where(np.abs(gradient) < np.float32(1e-15), 0,
                    gradient / (denominator + np.float32(1e-15)))
    deviation = math.sqrt(np.sum(denominator.astype(np.float64) * weak.astype(np.float64) ** 2) / len(targets))
    scale = np.float32(strength * deviation * len(targets) / (len(targets) + 1))
    noise = np.array([np.float32(_normal(feature, seed=seed, stream=1)) * scale for feature in range(5)])
    scores = _scores(bins, (gradient * factors).astype(np.float32), (denominator * factors).astype(np.float32),
                     features, borders, regularization, score, noise)
    winner = int(np.argmin(scores))
    if score == "NewtonCosine":
        wrong_weak = np.where(np.abs(gradient) < np.float32(1e-15), 0,
                              gradient / (weights + np.float32(1e-15)))
        wrong_deviation = math.sqrt(np.sum(weights.astype(np.float64) * wrong_weak.astype(np.float64) ** 2)
                                    / len(targets))
        wrong_scale = np.float32(strength * wrong_deviation * len(targets) / (len(targets) + 1))
        wrong_noise = np.array([np.float32(_normal(feature, seed=seed, stream=1)) * wrong_scale
                                for feature in range(5)])
        wrong_scores = _scores(bins, (gradient * factors).astype(np.float32),
                                (denominator * factors).astype(np.float32), features, borders,
                                regularization, score, wrong_noise)
        # Each fixture distinguishes Hessian-normalized score noise from the
        # plausible error of continuing to use original sample weights.
        assert winner != int(np.argmin(wrong_scores))
    sampling = {"bootstrap_type": kind, "random_seed": seed, "random_strength": strength}
    if kind in ("Bernoulli", "MVS"):
        sampling["subsample"] = 0.6
    if kind == "MVS":
        sampling["mvs_reg"] = 0.75
    if kind == "Bayesian":
        sampling["bagging_temperature"] = 1
    kwargs = options("Logloss", score, bias=bias, sample_weight=weights)
    actual = _native.train(bins, targets, features, borders, **kwargs, **sampling)
    assert actual.split_features[0, 0] == features[winner]
    assert actual.split_bins[0, 0] == borders[winner]
    # Force only the independently predicted winner into an unbootstrapped
    # oracle so that leaf fitting must use every original observation weight.
    expected = train_reference(bins, targets, features[winner:winner + 1], borders[winner:winner + 1], **kwargs)
    np.testing.assert_allclose(actual.leaf_weights, expected["leaf_weights"], rtol=5e-6, atol=5e-6)
    np.testing.assert_allclose(actual.leaf_values, expected["leaf_values"], rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("family", ["L2", "Cosine"])
@pytest.mark.parametrize("kind", ["No", "Bernoulli", "MVS"])
def test_rmse_first_order_and_newton_scores_are_equivalent(metal_device, family, kind):
    rng = np.random.default_rng(1927)
    bins = rng.integers(0, 5, (3, 257), dtype=np.uint8)
    y, weights = rng.normal(size=257).astype(np.float32), rng.uniform(0.2, 2, 257).astype(np.float32)
    features, borders = np.repeat(np.arange(3, dtype=np.uint32), 4), np.tile(np.arange(4, dtype=np.uint32), 3)
    sampling = {"bootstrap_type": kind, "random_seed": 812, "random_strength": 0.5}
    if kind != "No":
        sampling["subsample"] = 0.6
    first = _native.train(bins, y, features, borders,
                          **options("RMSE", family, iterations=3, depth=2, sample_weight=weights), **sampling)
    newton = _native.train(bins, y, features, borders,
                           **options("RMSE", "Newton" + family, iterations=3, depth=2, sample_weight=weights), **sampling)
    np.testing.assert_array_equal(newton.split_features, first.split_features)
    np.testing.assert_array_equal(newton.split_bins, first.split_bins)
    np.testing.assert_allclose(newton.predictions, first.predictions, rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("score", ["NewtonL2", "NewtonCosine"])
def test_public_classifier_newton_score_metadata_and_roundtrip(metal_device, tmp_path, score):
    rng = np.random.default_rng(7229)
    X = rng.normal(size=(257, 3)).astype(np.float32)
    y = np.where(X[:, 0] + X[:, 1] > 0.3, 13, -7)
    model = CatBoostMetalClassifier(iterations=3, depth=3, score_function=score, random_strength=0,
                                    leaf_estimation_iterations=3).fit(X, y)
    raw = model.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(X, prediction_type="RawFormulaVal", task_type="METAL"),
                               raw, rtol=5e-6, atol=5e-6)
    path = tmp_path / "newton.cbm"
    model.save_model(path)
    restored = CatBoostClassifier().load_model(str(path))
    assert restored.get_all_params()["score_function"] == score
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def test_sat_l2_public_gpu_predictions_match_standard_reader():
    rng = np.random.default_rng(483)
    x = rng.normal(size=(137, 3)).astype(np.float32)
    y = (x[:, 0] - .6 * x[:, 1]).astype(np.float32)
    model = CatBoostMetalRegressor(score_function="SatL2", iterations=4, depth=3).fit(x, y)
    np.testing.assert_allclose(model.predict(x, task_type="METAL"), model.predict(x), rtol=3e-6, atol=3e-7)
