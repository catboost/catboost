"""Greedy leaves use the CUDA scalar walker on resident Metal partitions."""
import ctypes as ct
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier

from catboost_metal import _greedy
from test_backtracking import cuda_leaf_walker
from test_greedy_training import route


CASES = [("RMSE", None, "Newton"), ("Logloss", None, "Newton"),
    ("CrossEntropy", None, "Gradient"), ("Poisson", None, "Newton"),
    ("Huber", .6, "Newton"), ("Expectile", .3, "Gradient"),
    ("Lq", 2.4, "Newton"), ("Tweedie", 1.5, "Gradient"),
    ("LogLinQuantile", .7, "Gradient"), ("Quantile", .7, "Gradient"),
    ("MAE", None, "Gradient"), ("MAPE", None, "Gradient")]


@pytest.fixture(autouse=True)
def actual_gpu_without_cpu_fit(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU fitting is forbidden")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier): monkeypatch.setattr(cls, "fit", forbidden)


def configure(session, mode):
    call = session._lib.cbm_greedy_session_set_backtracking
    call.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_char_p, ct.c_size_t]
    call.restype = ct.c_int
    error = ct.create_string_buffer(2048)
    result = call(session._handle, mode, error, len(error))
    if result: raise RuntimeError(error.value.decode())


@pytest.mark.parametrize("objective,param,method", CASES)
@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("mode", ["AnyImprovement", "Armijo"])
def test_weighted_greedy_backtracking_matches_independent_walker(objective, param, method, policy, mode):
    rng = np.random.default_rng(7881)
    bins = rng.integers(0, 4, (3, 71), dtype=np.uint8)
    targets = (.4 + .6 * (bins[0] > 1) + .25 * (bins[1] > 2)).astype(np.float32)
    if objective == "Logloss": targets = (bins[0] > 1).astype(np.float32)
    if objective == "CrossEntropy": targets = targets / 1.4
    weights = rng.uniform(.2, 1.8, len(targets)).astype(np.float32); weights[::9] = 0
    cf = np.repeat(np.arange(3, dtype=np.uint32), 3)
    cb = np.tile(np.arange(3, dtype=np.uint32), 3)
    baseline = np.linspace(-.1, .3, len(targets), dtype=np.float32)
    options = dict(grow_policy=policy, objective=objective, objective_param=param,
        score_function="L2", sample_weight=weights, iterations=2, depth=3,
        max_leaves=7, learning_rate=.3, l2_leaf_reg=.8, bias=0.,
        leaf_estimation_method=method, leaf_estimation_iterations=3, initial_predictions=baseline)
    with _greedy.TrainingSession(bins, targets, cf, cb, **options) as session:
        configure(session, 1 if mode == "AnyImprovement" else 2)
        prediction = baseline.copy()
        for _ in range(2):
            tree = session.step()
            ids = route(tree, bins)
            expected, expected_weights, _ = cuda_leaf_walker(targets, prediction, weights,
                ids, len(tree.leaf_values), objective=objective, objective_param=param,
                l2_leaf_reg=.8, leaf_estimation_method=method, leaf_estimation_iterations=3,
                leaf_estimation_backtracking=mode)
            np.testing.assert_allclose(tree.leaf_values, .3 * expected, rtol=2e-4, atol=2e-6)
            np.testing.assert_allclose(tree.leaf_weights, expected_weights, rtol=3e-6, atol=1e-6)
            prediction += tree.leaf_values[ids]
        np.testing.assert_array_equal(session.predictions(), prediction)
        with pytest.raises(RuntimeError, match="before the first tree"): configure(session, 0)


@pytest.mark.parametrize("mode,expected", [(1, 0.), (2, 100.)])
def test_equal_loss_and_armijo_acceptance(mode, expected):
    with _greedy.TrainingSession(np.zeros((1, 1), np.uint8), [100.], [], [],
            objective="Huber", objective_param=1., iterations=1, depth=0,
            learning_rate=1., l2_leaf_reg=.005, leaf_estimation_iterations=2) as session:
        configure(session, mode)
        session.step()
        np.testing.assert_array_equal(session.predictions(), [expected])


@pytest.mark.parametrize("mode", [1, 2])
def test_first_backtracking_acceptance_can_exceed_iteration_budget(mode):
    with _greedy.TrainingSession(np.zeros((1, 1), np.uint8), [1.], [], [],
            objective="Huber", objective_param=.1, iterations=1, depth=0,
            learning_rate=1., l2_leaf_reg=1e-6, leaf_estimation_iterations=2) as session:
        configure(session, mode)
        session.step()
        np.testing.assert_array_equal(session.predictions(), [np.float32(100000 / 65536)])


def test_invalid_backtracking_and_single_iteration_noop():
    options = dict(iterations=1, depth=1, leaf_estimation_iterations=1)
    with _greedy.TrainingSession(np.array([[0, 1]], np.uint8), [0., 1.], [0], [0], **options) as session:
        with pytest.raises(RuntimeError, match="Backtracking"): configure(session, 3)
        configure(session, 2)
        step = session.step()
    reference = _greedy.train(np.array([[0, 1]], np.uint8), [0., 1.], [0], [0], **options)
    np.testing.assert_array_equal(step.leaf_values, reference.trees[0].leaf_values)
