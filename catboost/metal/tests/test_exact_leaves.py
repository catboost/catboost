"""Exact weighted leaf optimization and host initialization regression checks.

Intentional corrections to the inspected CUDA source are covered explicitly:
full float32 residual ordering, convergent selection, no absolute weight epsilon,
safe empty leaves, full sorted host initialization, and original-target MAPE
weights. See cuda_scalar_reference.exact_leaf_value for the numerical oracle.
"""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor, _native
from catboost_metal._initialization import initialize_bias
from catboost_metal.regressor import quantize_features
from cuda_scalar_reference import exact_leaf_value, train_reference, weighted_loss


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Exact leaf checks must not invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Exact GPU leaf checks require Apple Silicon")
    return _native.device_info()


def options(objective="Quantile", alpha=0.5, **overrides):
    result = dict(iterations=1, depth=0, learning_rate=1, l2_leaf_reg=2, bias=0,
                  score_function="Cosine", objective=objective, objective_param=alpha,
                  leaf_estimation_method="Exact", leaf_estimation_iterations=1,
                  leaf_estimation_backtracking="No")
    result.update(overrides)
    return result


def train_constant(targets, weights=None, **kwargs):
    targets = np.asarray(targets, np.float32)
    empty = np.array([], np.uint32)
    return _native.train(np.zeros((1, targets.size), np.uint8), targets, empty, empty,
                         **options(**kwargs), sample_weight=weights)


@pytest.mark.parametrize("objective,targets,weights,parameters,expected", [
    ("RMSE", [0, 4, 10], [1, 3, 2], {}, 16 / 3),
    ("Logloss", [0, 1], [1, 3], {}, np.log(3)),
    ("CrossEntropy", [0, 0.5], [1, 3], {}, np.log(0.6)),
    ("Quantile", [0, 2], None, {}, 1e-6),
    ("Quantile", [0, 2, 4], None, {}, 2 - 1e-6),
    ("Quantile", [0, 2, 4], None, {"alpha": 0}, -1e-6),
    ("Quantile", [0, 2, 4], None, {"alpha": 1, "delta": 0}, 4),
    ("MAE", [0, 2], None, {}, 1e-6),
    ("MAE", [0, 2, 4], None, {}, 2 - 1e-6),
    # Private optimal-constant helper accepts delta; public MAE rejects params.
    ("MAE", [0, 2], None, {"delta": 0.002}, 0.002),
    ("MAPE", [10, 20], None, {}, 10),
    ("MAPE", [0, 0.5, 2], [1, 1, 3], {}, 0.5),
])
def test_initializer_matches_hand_derived_values(objective, targets, weights, parameters, expected):
    result = initialize_bias(targets, weights, objective, parameters)
    assert result == float(np.float32(expected))


@pytest.mark.parametrize("alpha,expected", [(0, -10), (0.5, 1), (1, 2)])
def test_initializer_zero_weight_endpoint_conventions(alpha, expected):
    assert initialize_bias([-10, 1, 2, 99], [0, 1, 1, 0], "Quantile",
                           {"alpha": alpha, "delta": 0}) == expected


def test_initializer_tiny_weights_do_not_trigger_an_absolute_epsilon():
    weights = np.array([1, 3, 1], np.float32) * np.float32(1e-30)
    assert initialize_bias([0, 1, 2], weights, "Quantile", {"delta": 0}) == 1


def test_initializer_wide_range_uses_observed_weighted_quantile():
    # CUDA's 100 value-space bisections return about 78,886,088 here.
    target = np.array([0] * 99 + [1e38], np.float32)
    assert initialize_bias(target, None, "Quantile", {"delta": 0}) == 0
    assert initialize_bias(target, None, "MAE", {}) == float(np.float32(1e-6))


def test_initializer_rmse_preserves_representable_mean_when_float_products_overflow():
    assert initialize_bias([1e20, 1e20], [1e20, 1e20], "RMSE", {}) == float(np.float32(1e20))


@pytest.mark.parametrize("targets,weights,objective,parameters", [
    ([], None, "Quantile", {}), ([1, 2], [0, 0], "Quantile", {}),
    ([1, 2], [1, -1], "Quantile", {}), ([1, np.nan], None, "MAE", {}),
    ([1, 2], [1], "MAPE", {}), ([0, 0], None, "Logloss", {}),
    ([1, 1], None, "CrossEntropy", {}), ([1, 2], None, "Poisson", {}),
])
def test_initializer_invalid_inputs_fail_cleanly(targets, weights, objective, parameters):
    with pytest.raises(ValueError):
        initialize_bias(targets, weights, objective, parameters)


@pytest.mark.parametrize("values,weights,alpha,expected", [
    ([3, 1, 2], [1, 4, 1], 0.5, 1),
    ([3, 1, 2], [1, 4, 1], 0.8, 2),
    ([-3, 0, 0, 7], [1, 1, 1, 1], 0.5, 0),
    ([-10, 1, 2, 99], [0, 1, 1, 0], 0, -10),
    ([-10, 1, 2, 99], [0, 1, 1, 0], 1, 2),
    ([1, 2, 3], [1, 1e-20, 0], 1, 2),
    ([1, 2], [0, 0], 0.5, 0),
])
def test_scalar_exact_reference_selection(values, weights, alpha, expected):
    assert exact_leaf_value(values, weights, "Quantile", alpha) == expected


@pytest.mark.parametrize("objective,alpha", [("Quantile", 0.3), ("MAE", 0.5), ("MAPE", 0.5)])
def test_exact_weighted_leaf_matches_sorted_scalar_reference(metal_device, objective, alpha):
    y = np.array([3, -1, 0.5, 2, 10, -10, 0.5], np.float32)
    weights = np.array([1, 3, 2, 1, 0, 0.25, 4], np.float32)
    expected = exact_leaf_value(y, weights, objective, alpha, targets=y)
    result = train_constant(y, weights, objective=objective, alpha=alpha)
    np.testing.assert_array_equal(result.predictions, expected)
    np.testing.assert_allclose(result.leaf_weights[0, 0], weights.sum())


def test_exact_sort_retains_all_float32_bits(metal_device):
    # These distinct values share CUDA's truncated radix key (bits10..32).
    values = np.array([1.0001219511032104, 1.0], np.float32)
    result = train_constant(values)
    np.testing.assert_array_equal(result.predictions, 1)


@pytest.mark.parametrize("rows,alpha,expected", [(65537, 0, 0), (65538, 1, 65536)])
def test_exact_selection_converges_beyond_sixteen_binary_search_steps(metal_device, rows, alpha, expected):
    weights = np.ones(rows, np.float32)
    if alpha == 1:
        weights[-1] = 0
    result = train_constant(np.arange(rows, dtype=np.float32), weights, alpha=alpha)
    np.testing.assert_array_equal(result.predictions, expected)


def test_exact_tiny_weights_preserve_weighted_median(metal_device):
    weights = np.array([1, 3, 1], np.float32) * np.float32(1e-30)
    result = train_constant([0, 1, 2], weights)
    np.testing.assert_array_equal(result.predictions, 1)


def test_exact_alpha_one_keeps_tiny_positive_weight_after_large_weight(metal_device):
    result = train_constant([1, 2, 3], np.array([1, 1e-20, 0], np.float32), alpha=1)
    np.testing.assert_array_equal(result.predictions, 2)


@pytest.mark.parametrize("bins,targets,weights", [
    ([[1]], [2], [1]),  # Empty first leaf and more leaves than documents.
    ([[0]], [2], [1]),
    ([[0, 1]], [2, 100], [1, 0]),  # Nonempty zero-weight leaf.
    ([[0, 0, 1], [0, 1, 1]], [-1, 1, 2], [1, 1, 1]),
])
def test_exact_empty_and_zero_weight_leaves_are_safe(metal_device, bins, targets, weights):
    bins, targets, weights = np.array(bins, np.uint8), np.array(targets, np.float32), np.array(weights, np.float32)
    features = np.arange(bins.shape[0], dtype=np.uint32)
    borders = np.zeros(features.size, np.uint32)
    kwargs = options(depth=4, sample_weight=weights)
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    np.testing.assert_allclose(actual.predictions, expected["predictions"], rtol=1e-6, atol=1e-7)
    assert np.all(actual.leaf_values[actual.leaf_weights == 0] == 0)


@pytest.mark.parametrize("backtracking", ["No", "AnyImprovement", "Armijo"])
def test_exact_leaf_solver_ignores_walker_damping_iterations_and_backtracking(metal_device, backtracking):
    y, weights = [3, -1, 2, 7], np.array([1, 3, 2, 1], np.float32)
    one = train_constant(y, weights, learning_rate=0.3)
    many = train_constant(y, weights, learning_rate=0.3, l2_leaf_reg=1000,
                          leaf_estimation_iterations=7, leaf_estimation_backtracking=backtracking)
    np.testing.assert_array_equal(many.predictions, one.predictions)
    np.testing.assert_array_equal(many.loss, one.loss)


def test_exact_mape_original_target_weights_fix_cuda_loss_increase(metal_device):
    y = np.array([10, 20], np.float32)
    # CUDA reweights residuals [-9,1], selecting +1 and worsening MAPE
    # from .475 to .5. Objective-correct original-target weights select -9.
    assert exact_leaf_value([-9, 1], [1, 1], "MAPE", targets=y) == -9
    result = train_constant(y, objective="MAPE", bias=19)
    np.testing.assert_array_equal(result.predictions, 10)
    np.testing.assert_allclose(result.loss, [0.475, 0.25], rtol=1e-6)
    assert result.loss[1] <= result.loss[0]


@pytest.mark.parametrize("objective,description,alpha", [
    ("Quantile", "Quantile:alpha=0.3;delta=0.002", 0.3),
    ("MAE", "MAE", 0.5), ("MAPE", "MAPE", 0.5),
])
def test_public_exact_multitree_fit_initialization_and_model_roundtrip(metal_device, tmp_path, objective, description, alpha):
    rng = np.random.default_rng(728)
    X = rng.normal(size=(257, 3)).astype(np.float32)
    y = (1 + X[:, 0] + 0.4 * (X[:, 1] > 0)).astype(np.float32)
    weights = (0.1 + 2 * rng.random(y.size)).astype(np.float32)
    weights[::17] = 0
    model = CatBoostMetalRegressor(loss_function=description, iterations=3, depth=2,
                                   border_count=7, learning_rate=0.3, l2_leaf_reg=2).fit(X, y, sample_weight=weights)
    parameters = {"alpha": alpha, "delta": 0.002} if objective == "Quantile" else {}
    assert model.bias_ == initialize_bias(y, weights, objective, parameters)
    _, bins, features, borders = quantize_features(X, 7)
    expected = train_reference(bins, y, features, borders,
                               **options(objective, alpha, iterations=3, depth=2, learning_rate=0.3,
                                         bias=model.bias_, sample_weight=weights))
    np.testing.assert_allclose(model.training_predictions_, expected["predictions"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(model.loss_history_, expected["loss"], rtol=1e-5, atol=1e-6)
    raw = model.predict(X)
    np.testing.assert_allclose(model.predict(X, task_type="METAL"), raw, rtol=5e-6, atol=5e-6)
    for format in ("cbm", "json"):
        path = tmp_path / f"exact-{objective}.{format}"
        model.save_model(path, format=format)
        restored = CatBoostRegressor().load_model(str(path), format=format)
        if format == "cbm":
            np.testing.assert_array_equal(restored.predict(X), raw)
        else:
            np.testing.assert_array_max_ulp(restored.predict(X), raw, maxulp=2)
        assert restored.get_all_params()["leaf_estimation_method"] == "Exact"
