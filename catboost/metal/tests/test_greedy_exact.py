"""Weighted Exact leaves for arbitrary Metal tree shapes, without CPU fitting."""
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier

from catboost_metal import _greedy
from cuda_scalar_reference import exact_leaf_value, weighted_loss
from test_greedy_training import route


@pytest.fixture(autouse=True)
def actual_gpu_no_cpu_fit(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs): raise AssertionError("CPU fitting is forbidden")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier): monkeypatch.setattr(cls, "fit", forbidden)


@pytest.mark.parametrize("objective,alpha", [("Quantile", 0.), ("Quantile", .23),
    ("Quantile", 1.), ("MAE", None), ("MAPE", None)])
@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("rows", [73, 8193])
def test_exact_weighted_leaves_match_independent_quantiles(objective, alpha, policy, rows):
    rng = np.random.default_rng(885)
    bins = rng.integers(0, 4, (2, rows), dtype=np.uint8)
    y = (4 * bins[0].astype(float) - 7 + rng.normal(0, 1, rows)).astype(np.float32)
    y[::17] = 0
    weights = rng.uniform(.1, 2, rows).astype(np.float32); weights[::5] = 0
    initial = rng.uniform(-5, 5, rows).astype(np.float32)
    cf = np.repeat(np.arange(2, dtype=np.uint32), 3); cb = np.tile(np.arange(3, dtype=np.uint32), 2)
    result = _greedy.train(bins, y, cf, cb, grow_policy=policy, objective=objective,
        objective_param=alpha, sample_weight=weights, initial_predictions=initial,
        iterations=2, depth=4, max_leaves=9, learning_rate=.3, l2_leaf_reg=14.,
        leaf_estimation_method="Exact", leaf_estimation_iterations=17,
        leaf_estimation_backtracking="Armijo", score_function="L2")
    prediction = initial.copy()
    for tree in result.trees:
        ids = route(tree, bins)
        residuals = (y - prediction).astype(np.float32)
        expected = np.array([exact_leaf_value(residuals[ids == leaf], weights[ids == leaf],
            objective, .5 if alpha is None else alpha, targets=y[ids == leaf])
            for leaf in range(len(tree.leaf_values))], np.float32) * np.float32(.3)
        np.testing.assert_array_equal(tree.leaf_values, expected)
        np.testing.assert_allclose(tree.leaf_weights,
            np.bincount(ids, weights=weights.astype(float), minlength=len(expected)), rtol=3e-6, atol=2e-5)
        prediction += expected[ids]
        assert tree.loss == pytest.approx(weighted_loss(y, prediction, weights, objective, alpha), rel=3e-6)
    np.testing.assert_array_equal(result.predictions, prediction)


@pytest.mark.parametrize("objective,alpha,targets,weights,initial", [
    ("Quantile", 0., [-10., 1., 2., 99.], [0., 1., 1., 0.], [0., 0., 0., 0.]),
    ("Quantile", 1., [-10., 1., 2., 99.], [0., 1., 1., 0.], [0., 0., 0., 0.]),
    ("Quantile", .5, [0., 1e-8, 2e-8], [1., 1., 1.], [0., 0., 0.]),
    ("MAE", None, [0., 1., 2.], [1e-30, 3e-30, 1e-30], [0., 0., 0.]),
    ("MAPE", None, [1., 10., 100.], [1., 1., 1.], [0., 9., 0.]),
])
def test_exact_endpoint_tiny_weight_and_original_target_mape_cases(objective, alpha, targets, weights, initial):
    targets, weights, initial = [np.array(value, np.float32) for value in (targets, weights, initial)]
    expected = exact_leaf_value(targets - initial, weights, objective,
                               .5 if alpha is None else alpha, targets=targets)
    result = _greedy.train(np.zeros((1, len(targets)), np.uint8), targets, [], [],
        sample_weight=weights, initial_predictions=initial, objective=objective,
        objective_param=alpha, leaf_estimation_method="Exact", iterations=1, depth=0,
        learning_rate=1.)
    np.testing.assert_array_equal(result.trees[0].leaf_values, [expected])
    np.testing.assert_array_equal(result.predictions, initial + np.float32(expected))


def test_exact_supports_empty_leaves_in_deep_lossguide():
    # Hessian-based scoring has defined zero-child gains for Quantile. Lossguide
    # can grow empty siblings, which must get zero values and original weights.
    result = _greedy.train(np.zeros((1, 19), np.uint8), np.arange(19), [0], [0],
        objective="Quantile", objective_param=.5, leaf_estimation_method="Exact",
        grow_policy="Lossguide", depth=100, max_leaves=31, iterations=1,
        score_function="NewtonL2", learning_rate=1.)
    assert len(result.trees[0].leaf_values) == 31
    np.testing.assert_array_equal(result.trees[0].leaf_values, np.r_[9., np.zeros(30)])
    np.testing.assert_array_equal(result.trees[0].leaf_weights, np.r_[19., np.zeros(30)])
