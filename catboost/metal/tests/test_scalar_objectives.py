"""CUDA pointwise-loss and Gradient/Newton checks without CPU training."""

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
        pytest.fail("Pointwise CUDA-port checks cannot train CPU CatBoost")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def options(objective, objective_param=None, **overrides):
    result = dict(iterations=1, depth=2, learning_rate=0.5, l2_leaf_reg=2, bias=0,
                  score_function="Cosine", objective=objective, objective_param=objective_param,
                  leaf_estimation_method="Newton", leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No")
    result.update(overrides)
    return result


def compare(bins, targets, features, borders, **kwargs):
    reference = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, reference["depths"])
    for tree, depth in enumerate(actual.depths):
        np.testing.assert_array_equal(actual.split_features[tree, :depth], reference["split_features"][tree, :depth])
        np.testing.assert_array_equal(actual.split_bins[tree, :depth], reference["split_bins"][tree, :depth])
    for name in ("leaf_values", "leaf_weights", "predictions"):
        np.testing.assert_allclose(getattr(actual, name), reference[name], rtol=8e-5, atol=8e-5, err_msg=name)
    np.testing.assert_allclose(actual.loss, reference["loss"], rtol=8e-5, atol=8e-6)
    assert np.isfinite(actual.predictions).all()
    assert np.isfinite(actual.loss).all()
    assert actual.stats["kernel_dispatches"] > 0
    return actual, reference


HAND_CASES = [
    ("Poisson", None, "Newton", [0, 2, 4, 6], np.log(2), [-0.1, 1 / 3]),
    ("Poisson", None, "Gradient", [0, 2, 4, 6], np.log(2), [-1 / 6, 0.5]),
    ("Huber", 1, "Newton", [-2, 0.5, 3, 0], 0, [0.05, 0.5]),
    ("Huber", 1, "Gradient", [-2, 0.5, 3, 0], 0, [1 / 24, 0.25]),
    ("Expectile", 0.25, "Newton", [-2, 0.5, 3, 0], 0, [-0.225, 0.5]),
    ("Expectile", 0.25, "Gradient", [-2, 0.5, 3, 0], 0, [-0.1875, 0.375]),
]


@pytest.mark.parametrize("objective,param,method,targets,bias,expected", HAND_CASES)
def test_scalar_oracle_matches_hand_derived_weighted_leaf_steps(objective, param, method, targets, bias, expected):
    reference = train_reference(np.array([[0, 0, 1, 1]], np.uint8), np.array(targets, np.float32),
                                 np.array([0], np.uint32), np.array([0], np.uint32),
                                 **options(objective, param, bias=bias, leaf_estimation_method=method,
                                           sample_weight=np.array([1, 3, 2, 0], np.float32)))
    np.testing.assert_allclose(reference["leaf_values"][0, :2], expected, rtol=5e-7, atol=1e-8)


@pytest.mark.parametrize("objective,param,method,targets,bias,expected", HAND_CASES)
def test_gpu_matches_hand_derived_weighted_leaf_steps(metal_device, objective, param, method, targets, bias, expected):
    result, _ = compare(np.array([[0, 0, 1, 1]], np.uint8), np.array(targets, np.float32),
                         np.array([0], np.uint32), np.array([0], np.uint32),
                         **options(objective, param, bias=bias, leaf_estimation_method=method,
                                   sample_weight=np.array([1, 3, 2, 0], np.float32)))
    np.testing.assert_allclose(result.leaf_values[0, :2], expected, rtol=5e-6, atol=1e-7)


def test_scalar_oracle_huber_and_expectile_branch_boundaries():
    loss, gradient, hessian = objective_terms(np.array([-1, -0.5, 0, 0.5, 1]), 0, "Huber", 1)
    np.testing.assert_array_equal(loss, [0.5, 0.125, 0, 0.125, 0.5])
    np.testing.assert_array_equal(gradient, [-1, -0.5, 0, 0.5, 1])
    np.testing.assert_array_equal(hessian, [0, 1, 1, 1, 0])
    _, gradient, hessian = objective_terms(np.array([-1, 0, 1]), 0, "Expectile", 0.25)
    np.testing.assert_array_equal(gradient, [-1.5, 0, 0.5])
    np.testing.assert_array_equal(hessian, [1.5, 1.5, 0.5])


@pytest.mark.parametrize("objective,param,description", [
    ("Poisson", None, "Poisson"), ("Huber", 0.8, "Huber:delta=0.8"),
    ("Huber", 0, "Huber:delta=0"), ("Expectile", 0.3, "Expectile:alpha=0.3"),
    ("Expectile", 0, "Expectile:alpha=0"), ("Expectile", 1, "Expectile:alpha=1"),
])
def test_scalar_oracle_loss_matches_upstream_metric_without_training(objective, param, description):
    from catboost.utils import eval_metric
    targets = np.array([0.5, 2, 4, 6], np.float32)
    raw = np.array([-0.5, 0.2, 2, 3], np.float32)
    weights = np.array([1, 3, 2, 0], np.float32)
    expected = eval_metric(targets, raw, description, weight=weights)[0]
    np.testing.assert_allclose(weighted_loss(targets, raw, weights, objective, param),
                               expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize("method,expected", [("Newton", 1), ("Gradient", 1 / 3)])
def test_huber_has_zero_newton_curvature_at_exact_delta(metal_device, method, expected):
    empty = np.array([], np.uint32)
    result, _ = compare(np.zeros((1, 2), np.uint8), np.array([-1, 1], np.float32), empty, empty,
                         **options("Huber", 1, depth=0, learning_rate=1, leaf_estimation_method=method,
                                   sample_weight=np.array([1, 3], np.float32)))
    np.testing.assert_allclose(result.predictions, expected, rtol=1e-6)


def test_expectile_zero_residual_uses_one_minus_alpha_curvature(metal_device):
    empty = np.array([], np.uint32)
    result, _ = compare(np.zeros((1, 2), np.uint8), np.array([0, 1], np.float32), empty, empty,
                         **options("Expectile", 0.25, depth=0, learning_rate=1, l2_leaf_reg=0.5,
                                   sample_weight=np.array([2, 1], np.float32)))
    np.testing.assert_allclose(result.predictions, 0.125, rtol=1e-6)


@pytest.mark.parametrize("objective,param", [("Poisson", None), ("Huber", 0.8), ("Expectile", 0.3)])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("steps", [1, 3])
@pytest.mark.parametrize("score_function", ["L2", "Cosine"])
def test_weighted_multifeature_iterative_training_matches_cuda_oracle(
        metal_device, objective, param, method, steps, score_function):
    rng = np.random.default_rng(3721)
    bins = rng.integers(0, 6, size=(3, 257), dtype=np.uint8)
    signal = (0.6 * (bins[1] > 2) - 0.3 * (bins[2] > 3)
              + rng.normal(0, 0.07, 257))
    targets = np.exp(signal) if objective == "Poisson" else signal
    weights = (0.2 + rng.random(257) * 3).astype(np.float32)
    weights[::17] = 0
    result, _ = compare(bins, targets.astype(np.float32),
                         np.repeat(np.arange(3, dtype=np.uint32), 5),
                         np.tile(np.arange(5, dtype=np.uint32), 3),
                         **options(objective, param, iterations=3, depth=2, learning_rate=0.15,
                                   sample_weight=weights, leaf_estimation_method=method,
                                   leaf_estimation_iterations=steps, score_function=score_function))
    assert result.loss[-1] < result.loss[0]


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
def test_gradient_leaf_estimation_also_applies_to_existing_objectives(metal_device, objective):
    targets = {"RMSE": [-1, 0, 2, 3], "Logloss": [0, 0, 1, 1],
               "CrossEntropy": [0.1, 0.3, 0.6, 0.9]}[objective]
    compare(np.array([[0, 0, 1, 1]], np.uint8), np.array(targets, np.float32),
             np.array([0], np.uint32), np.array([0], np.uint32),
             **options(objective, iterations=3, leaf_estimation_method="Gradient",
                       leaf_estimation_iterations=3, sample_weight=np.array([1, 3, 2, 0], np.float32)))


@pytest.mark.parametrize("objective,param", [("Expectile", 0.5), ("Huber", 1000)])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_quadratic_objective_special_cases_share_rmse_training(metal_device, objective, param, method):
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    targets = np.array([-1, 0.5, 2, 3], np.float32)
    candidates = np.array([0], np.uint32)
    kwargs = dict(iterations=3, leaf_estimation_method=method, leaf_estimation_iterations=3,
                  sample_weight=np.array([1, 3, 2, 0], np.float32))
    rmse = _native.train(bins, targets, candidates, candidates, **options("RMSE", **kwargs))
    quadratic = _native.train(bins, targets, candidates, candidates, **options(objective, param, **kwargs))
    np.testing.assert_array_equal(quadratic.predictions, rmse.predictions)
    np.testing.assert_array_equal(quadratic.leaf_values, rmse.leaf_values)
    np.testing.assert_allclose(quadratic.loss, 0.5 * rmse.loss.astype(np.float64) ** 2,
                               rtol=1e-6, atol=1e-8)


def test_valid_poisson_loss_can_be_negative(metal_device):
    empty = np.array([], np.uint32)
    result, _ = compare(np.zeros((1, 3), np.uint8), np.array([8.5, 10.5, 12], np.float32), empty, empty,
                         **options("Poisson", depth=0, bias=np.log(10),
                                   sample_weight=np.array([1, 2, 3], np.float32)))
    assert np.all(result.loss < 0)


@pytest.mark.parametrize("objective,param", [("Huber", 0), ("Expectile", 0), ("Expectile", 1)])
def test_valid_loss_parameter_endpoints(metal_device, objective, param):
    compare(np.array([[0, 0, 1, 1]], np.uint8), np.array([-2, 0, 1, 3], np.float32),
             np.array([0], np.uint32), np.array([0], np.uint32),
             **options(objective, param, leaf_estimation_iterations=3))


@pytest.mark.parametrize("objective,param", [("Poisson", None), ("Huber", 1), ("Expectile", 0.7)])
def test_zero_weight_large_target_has_no_effect(metal_device, objective, param):
    bins = np.array([[0, 0, 1, 1]], np.uint8)
    candidates = np.array([0], np.uint32)
    kwargs = options(objective, param, sample_weight=np.array([1, 2, 3, 0], np.float32))
    actual, _ = compare(bins, np.array([0.5, 1, 2, 1e20], np.float32), candidates, candidates, **kwargs)
    clean = _native.train(bins, np.array([0.5, 1, 2, 0], np.float32), candidates, candidates, **kwargs)
    np.testing.assert_array_equal(actual.predictions, clean.predictions)
    np.testing.assert_array_equal(actual.loss, clean.loss)


@pytest.mark.parametrize("objective,param,description", [
    ("Poisson", None, "Poisson"), ("Huber", 0.8, "Huber:delta=0.8"),
    ("Expectile", 0.3, "Expectile:alpha=0.3"),
])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_public_loss_description_model_roundtrip_and_gpu_prediction(
        metal_device, tmp_path, objective, param, description, method):
    rng = np.random.default_rng(657)
    X = rng.normal(size=(133, 3)).astype(np.float32)
    signal = 0.5 * X[:, 0] + 0.2 * (X[:, 2] > 0)
    y = (np.exp(signal) if objective == "Poisson" else signal).astype(np.float32)
    weights = np.linspace(0, 3, len(y), dtype=np.float32)
    model = CatBoostMetalRegressor(loss_function=description, iterations=3, depth=2,
                                   learning_rate=0.15, l2_leaf_reg=2, border_count=7,
                                   leaf_estimation_method=method, leaf_estimation_iterations=3).fit(
                                       X, y, sample_weight=weights)
    assert model.bias_ == 0
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, y, features, borders,
                               **options(objective, param, iterations=3, depth=2, learning_rate=0.15,
                                         sample_weight=weights, leaf_estimation_method=method,
                                         leaf_estimation_iterations=3))
    raw = model.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(raw, expected["predictions"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(model.loss_history_, expected["loss"], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(model.predict(X, prediction_type="RawFormulaVal", task_type="METAL"),
                               raw, rtol=5e-6, atol=5e-6)
    for format in ("cbm", "json"):
        path = tmp_path / f"{objective}-{method}.{format}"
        model.save_model(path, format=format)
        restored = CatBoostRegressor().load_model(str(path), format=format)
        restored_raw = restored.predict(X, prediction_type="RawFormulaVal")
        if format == "cbm":
            np.testing.assert_array_equal(restored_raw, raw)
        else:
            # Upstream JSON serialization can round a decimal leaf by one
            # float64 ULP; CBM remains an exact binary roundtrip.
            np.testing.assert_array_max_ulp(restored_raw, raw, maxulp=2)
        np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-12, atol=1e-12)
        assert restored.get_all_params()["leaf_estimation_method"] == method
    if objective == "Poisson":
        np.testing.assert_allclose(model.predict(X), np.exp(raw), rtol=1e-12)
        np.testing.assert_allclose(model.predict(X, task_type="METAL"), np.exp(raw), rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("description,method,steps", [
    ("Poisson", "Newton", 10), ("Poisson", "Gradient", 1),
    ("Huber:delta=1", "Newton", 1), ("Huber:delta=1", "Gradient", 1),
    ("Expectile:alpha=0.3", "Newton", 5), ("Expectile:alpha=0.3", "Gradient", 10),
])
def test_public_leaf_defaults_match_cuda_options(description, method, steps):
    model = CatBoostMetalRegressor(loss_function=description, leaf_estimation_method=method)
    assert model.leaf_estimation_iterations == steps
    assert model.boost_from_average is False


@pytest.mark.parametrize("description", ["Huber", "Huber:delta=-1", "Huber:delta=nan",
                                         "Expectile", "Expectile:alpha=-0.1", "Expectile:alpha=1.1",
                                         "Poisson:delta=1", "Huber:delta=1;unknown=2"])
def test_invalid_loss_description_rejected_before_gpu(description):
    with pytest.raises((TypeError, ValueError)):
        CatBoostMetalRegressor(loss_function=description)


@pytest.mark.parametrize("description", ["Poisson", "Huber:delta=1", "Expectile:alpha=0.3"])
def test_new_losses_reject_unsupported_boost_from_average(description):
    with pytest.raises(ValueError):
        CatBoostMetalRegressor(loss_function=description, boost_from_average=True)


@pytest.mark.parametrize("objective,param", [("Huber", -1), ("Huber", np.nan), ("Huber", np.inf),
                                           ("Expectile", -0.1), ("Expectile", 1.1), ("Expectile", np.nan)])
def test_native_invalid_objective_parameters_rejected_before_gpu(monkeypatch, objective, param):
    monkeypatch.setattr(_native, "build_library", lambda: pytest.fail("Reached native runtime"))
    with pytest.raises(ValueError):
        _native.train(np.array([[0, 1]], np.uint8), np.array([0, 1], np.float32),
                       np.array([0], np.uint32), np.array([0], np.uint32), **options(objective, param))
