"""Remaining scalar CUDA objectives; Exact leaf selection has separate tests."""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor, _native
from catboost_metal.regressor import quantize_features
from cuda_scalar_reference import objective_terms, train_reference, weighted_loss


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Additional objective checks must not train CPU CatBoost")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def options(objective, param=None, method="Gradient", **overrides):
    result = dict(iterations=1, depth=2, learning_rate=0.5, l2_leaf_reg=2, bias=0,
                  score_function="Cosine", objective=objective, objective_param=param,
                  leaf_estimation_method=method, leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No")
    result.update(overrides)
    return result


def compare(bins, targets, features, borders, **kwargs):
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    for i, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[i, :depth], expected["split_features"][i, :depth])
        np.testing.assert_array_equal(actual.split_bins[i, :depth], expected["split_bins"][i, :depth])
    for key in ("leaf_values", "leaf_weights", "predictions"):
        np.testing.assert_allclose(getattr(actual, key), expected[key], rtol=1e-4, atol=1e-4, err_msg=key)
    np.testing.assert_allclose(actual.loss, expected["loss"], rtol=1e-4, atol=1e-5)
    assert np.isfinite(actual.loss).all()
    assert actual.stats["kernel_dispatches"] > 0
    return actual


HAND_CASES = [
    ("Lq", 1.5, "Gradient", [-4, 0, 1, 9], [-0.25, 0.375]),
    ("Lq", 3, "Newton", [-4, 0, 1, 9], [-12 / 13, 3 / 14]),
    ("Tweedie", 1.5, "Newton", [0, 2, 4, 6], [1 / 7, 3 / 7]),
    ("Tweedie", 1.5, "Gradient", [0, 2, 4, 6], [1 / 6, 0.75]),
    ("LogLinQuantile", 0.25, "Gradient", [0, 0, 4, 6], [-0.25, 0.0625]),
]


@pytest.mark.parametrize("objective,param,method,targets,expected", HAND_CASES)
def test_scalar_reference_matches_hand_derived_updates(objective, param, method, targets, expected):
    result = train_reference(np.array([[0, 0, 1, 1]], np.uint8), np.array(targets, np.float32),
                              np.array([0], np.uint32), np.array([0], np.uint32),
                              **options(objective, param, method, sample_weight=np.array([1, 3, 2, 0], np.float32)))
    np.testing.assert_allclose(result["leaf_values"][0, :2], expected, rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("objective,param,method,targets,expected", HAND_CASES)
def test_gpu_matches_hand_derived_updates(metal_device, objective, param, method, targets, expected):
    result = compare(np.array([[0, 0, 1, 1]], np.uint8), np.array(targets, np.float32),
                      np.array([0], np.uint32), np.array([0], np.uint32),
                      **options(objective, param, method, sample_weight=np.array([1, 3, 2, 0], np.float32)))
    np.testing.assert_allclose(result.leaf_values[0, :2], expected, rtol=5e-6, atol=1e-7)


@pytest.mark.parametrize("objective,param,description", [
    ("Lq", 1.5, "Lq:q=1.5"), ("Lq", 3, "Lq:q=3"),
    ("Tweedie", 1.2, "Tweedie:variance_power=1.2"),
    ("Tweedie", 1.8, "Tweedie:variance_power=1.8"),
    ("LogLinQuantile", 0.3, "LogLinQuantile:alpha=0.3"),
    ("Quantile", 0.3, "Quantile:alpha=0.3;delta=0"), ("MAPE", None, "MAPE"),
])
def test_scalar_reference_loss_agrees_with_upstream_metric(objective, param, description):
    from catboost.utils import eval_metric
    y = np.array([0.5, 2, 4, 6], np.float32)
    raw = np.array([-0.5, 0.2, 2, 3], np.float32)
    weights = np.array([1, 3, 2, 0], np.float32)
    expected = eval_metric(y, raw, description, weight=weights)[0]
    np.testing.assert_allclose(weighted_loss(y, raw, weights, objective, param), expected,
                               rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("objective,param,method", [
    ("Lq", 1, "Gradient"), ("Lq", 1.5, "Gradient"), ("Lq", 2, "Newton"),
    ("Lq", 3, "Newton"), ("Lq", 3, "Gradient"),
    ("Tweedie", 1.2, "Newton"), ("Tweedie", 1.5, "Newton"),
    ("Tweedie", 1.8, "Gradient"), ("LogLinQuantile", 0.3, "Gradient"),
    ("Quantile", 0.3, "Gradient"), ("MAE", None, "Gradient"), ("MAPE", None, "Gradient"),
])
def test_multitree_weighted_updates_match_reference(metal_device, objective, param, method):
    rng = np.random.default_rng(10429)
    bins = rng.integers(0, 8, (3, 259), dtype=np.uint8)
    signal = 0.4 * (bins[0] > 3) - 0.2 * (bins[2] > 5) + rng.normal(0, 0.08, 259)
    targets = (np.exp(signal) if objective in ("Tweedie", "LogLinQuantile", "MAPE")
               else signal).astype(np.float32)
    weights = (0.25 + rng.random(259) * 2).astype(np.float32)
    weights[::19] = 0
    compare(bins, targets, np.repeat(np.arange(3, dtype=np.uint32), 7),
             np.tile(np.arange(7, dtype=np.uint32), 3),
             **options(objective, param, method, iterations=3, leaf_estimation_iterations=3,
                       learning_rate=0.15, sample_weight=weights))


@pytest.mark.parametrize("objective,param,target,gradient", [
    ("Lq", 1, 0, -1), ("LogLinQuantile", 0.25, 1, -0.75),
    ("Quantile", 0.25, 0, -0.75), ("MAE", None, 0, -0.5), ("MAPE", None, 0, -1),
])
def test_exact_residual_equality_preserves_cuda_negative_gradient_branch(
        metal_device, objective, param, target, gradient):
    empty = np.array([], np.uint32)
    _, reference_gradient, _ = objective_terms(np.array([target]), 0, objective, param)
    np.testing.assert_array_equal(reference_gradient, gradient)
    result = compare(np.zeros((1, 1), np.uint8), np.array([target], np.float32), empty, empty,
                      **options(objective, param, depth=0, learning_rate=1))
    np.testing.assert_allclose(result.predictions, gradient / 3, rtol=1e-6, atol=1e-7)


def test_mae_reports_twice_quantile_loss_without_doubling_derivatives(metal_device):
    bins, y = np.array([[0, 0, 1, 1]], np.uint8), np.array([-2, 0.5, 1, 4], np.float32)
    candidates = np.array([0], np.uint32)
    kwargs = dict(iterations=3, leaf_estimation_iterations=2, sample_weight=np.array([1, 3, 2, 0], np.float32))
    mae = _native.train(bins, y, candidates, candidates, **options("MAE", **kwargs))
    quantile = _native.train(bins, y, candidates, candidates, **options("Quantile", 0.5, **kwargs))
    np.testing.assert_array_equal(mae.predictions, quantile.predictions)
    np.testing.assert_array_equal(mae.leaf_values, quantile.leaf_values)
    np.testing.assert_allclose(mae.loss, quantile.loss * 2, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("description,method,steps", [
    ("Lq:q=1", "Gradient", 1), ("Lq:q=1.5", "Gradient", 1), ("Lq:q=2", "Newton", 1),
    ("Tweedie:variance_power=1.5", "Newton", 20), ("LogLinQuantile", "Gradient", 1),
    ("Quantile", "Exact", 1), ("MAE", "Exact", 1), ("MAPE", "Exact", 1),
])
def test_public_default_leaf_methods_follow_cuda(description, method, steps):
    model = CatBoostMetalRegressor(loss_function=description)
    assert model.leaf_estimation_method == method
    assert model.leaf_estimation_iterations == steps


@pytest.mark.parametrize("description,method", [
    ("Lq:q=1.5", "Newton"), ("Lq:q=2", "Exact"),
    ("Tweedie:variance_power=1.5", "Exact"), ("LogLinQuantile", "Newton"),
    ("LogLinQuantile", "Exact"), ("Quantile", "Newton"), ("MAE", "Newton"), ("MAPE", "Newton"),
])
def test_incompatible_leaf_method_rejected_before_gpu(description, method):
    with pytest.raises(ValueError):
        CatBoostMetalRegressor(loss_function=description, leaf_estimation_method=method)


@pytest.mark.parametrize("description", ["Lq", "Lq:q=0.9", "Lq:q=inf", "Tweedie",
                                         "Tweedie:variance_power=1", "Tweedie:variance_power=2",
                                         "LogLinQuantile:alpha=-0.1", "Quantile:delta=0.02",
                                         "MAE:alpha=0.5", "MAE:delta=0.002"])
def test_invalid_additional_loss_description_rejected_before_gpu(description):
    with pytest.raises((TypeError, ValueError)):
        CatBoostMetalRegressor(loss_function=description)


@pytest.mark.parametrize("objective,param,method,description", [
    ("Lq", 1.5, "Gradient", "Lq:q=1.5"), ("Lq", 3, "Newton", "Lq:q=3"),
    ("Tweedie", 1.5, "Newton", "Tweedie:variance_power=1.5"),
    ("LogLinQuantile", 0.3, "Gradient", "LogLinQuantile:alpha=0.3"),
])
def test_public_additional_loss_model_roundtrip(metal_device, tmp_path, objective, param, method, description):
    rng = np.random.default_rng(927)
    X = rng.normal(size=(259, 3)).astype(np.float32)
    signal = 0.3 * X[:, 0] + 0.2 * (X[:, 1] > 0)
    targets = (np.exp(signal) if objective in ("Tweedie", "LogLinQuantile") else signal).astype(np.float32)
    weights = np.linspace(0, 3, len(targets), dtype=np.float32)
    model = CatBoostMetalRegressor(loss_function=description, iterations=3, depth=2,
                                   border_count=7, learning_rate=0.15, l2_leaf_reg=2,
                                   leaf_estimation_iterations=3).fit(X, targets, sample_weight=weights)
    assert model.bias_ == 0
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, targets, features, borders,
                               **options(objective, param, method, iterations=3, depth=2,
                                         learning_rate=0.15, sample_weight=weights,
                                         leaf_estimation_iterations=3))
    raw = model.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(raw, expected["predictions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(model.loss_history_, expected["loss"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(model.predict(X, prediction_type="RawFormulaVal", task_type="METAL"),
                               raw, rtol=5e-6, atol=5e-6)
    for format in ("cbm", "json"):
        path = tmp_path / f"additional-{objective}-{method}.{format}"
        model.save_model(path, format=format)
        restored = CatBoostRegressor().load_model(str(path), format=format)
        if format == "cbm":
            np.testing.assert_array_equal(restored.predict(X, prediction_type="RawFormulaVal"), raw)
        else:
            np.testing.assert_array_max_ulp(restored.predict(X, prediction_type="RawFormulaVal"), raw, maxulp=2)
        np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-14, atol=1e-15)
    if objective == "Tweedie":
        np.testing.assert_allclose(model.predict(X), np.exp(raw), rtol=1e-14)
        np.testing.assert_allclose(model.predict(X, task_type="METAL"), np.exp(raw), rtol=5e-6, atol=5e-6)
