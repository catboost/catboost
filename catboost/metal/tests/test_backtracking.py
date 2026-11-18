"""CUDA leaf-walker acceptance rules exercised through real Metal sessions.

Sources: cuda/methods/leaves_estimation/{descent_helpers,step_estimator}.cpp.
The independent reference projects audited target formulas in NumPy; it neither
calls a trainer nor imports the Metal optimizer implementation.
"""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import _native
from cuda_scalar_reference import objective_terms, train_reference, weighted_loss


MODES = ("AnyImprovement", "Armijo")
OBJECTIVES = (("RMSE", None), ("Logloss", None), ("CrossEntropy", None),
              ("Poisson", None), ("Huber", 0.8), ("Expectile", 0.3))


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Backtracking tests must not invoke CPU CatBoost training")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def options(**overrides):
    result = dict(iterations=1, depth=0, learning_rate=1, l2_leaf_reg=2,
                  bias=0, score_function="Cosine", objective="RMSE",
                  leaf_estimation_iterations=4, leaf_estimation_method="Newton",
                  leaf_estimation_backtracking="AnyImprovement")
    result.update(overrides)
    return result


def cuda_leaf_walker(targets, baseline, weights, leaf_ids, leaves, *, objective,
                     objective_param=None, l2_leaf_reg, leaf_estimation_method,
                     leaf_estimation_iterations, leaf_estimation_backtracking):
    """CUDA's global scalar line search, with float points/directions.

    Values are negative UNNORMALIZED objective sums. Neither L2 ridge loss nor
    ridge gradient is enabled: lambda only damps the diagonal. RMSE's objective
    has no factor 1/2. Every trial counts; before the first accepted point the
    walker permits at least 100 attempts. Equal objective values are accepted.
    """
    targets = np.asarray(targets, np.float32)
    baseline = np.asarray(baseline, np.float32)
    weights = np.asarray(weights, np.float32)
    active = weights > 0
    leaf_ids = np.asarray(leaf_ids, np.int64)
    member_ids = leaf_ids[active]
    leaf_weights = np.bincount(leaf_ids, weights=weights.astype(np.float64), minlength=leaves)
    regularization = float(np.float32(l2_leaf_reg)) or float(np.float32(1e-20))
    point = np.zeros(leaves, np.float32)
    trace = []

    def project(candidate):
        raw = (baseline + candidate[leaf_ids]).astype(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            loss, derivative, curvature = objective_terms(
                targets[active], raw[active], objective, objective_param)
            # CUDA emits float per-row statistics, then widens partition sums.
            g = (weights[active].astype(np.float64) * derivative).astype(np.float32)
            h = (weights[active].astype(np.float64) * curvature).astype(np.float32)
            row_score = -(weights[active] * loss.astype(np.float32))
        value = row_score.sum(dtype=np.float64)
        gradient = np.bincount(member_ids, weights=g.astype(np.float64), minlength=leaves)
        hessian = np.bincount(member_ids, weights=h.astype(np.float64), minlength=leaves)
        return value, gradient, hessian

    def direction(gradient, hessian):
        diagonal = (leaf_weights if leaf_estimation_method == "Gradient" else hessian) + regularization
        result = np.zeros(leaves, np.float32)
        positive = diagonal > 0
        result[positive] = (gradient[positive] / (diagonal[positive] + 1e-20)).astype(np.float32)
        return result

    def trial(step, move):
        candidate = (point.astype(np.float64) + step * move.astype(np.float64)).astype(np.float32)
        candidate[leaf_weights < 1e-20] = 0
        return candidate

    value, gradient, hessian = project(point)
    if leaf_estimation_backtracking == "No" or leaf_estimation_iterations == 1:
        for _ in range(leaf_estimation_iterations):
            point = trial(1, direction(gradient, hessian))
            value, gradient, hessian = project(point)
            trace.append((1.0, True, value))
        return point, leaf_weights, trace

    attempts = 0
    updated = False
    while attempts < leaf_estimation_iterations:
        move = direction(gradient, hessian)
        direction_dot = np.dot(gradient, move.astype(np.float64))
        step = 1.0
        while attempts < leaf_estimation_iterations or (not updated and attempts < 100):
            candidate = trial(step, move)
            candidate_value, candidate_gradient, candidate_hessian = project(candidate)
            threshold = value
            if leaf_estimation_backtracking == "Armijo":
                threshold += 1e-5 * step * direction_dot
            accepted = np.isfinite(candidate_value) and candidate_value >= threshold
            trace.append((step, bool(accepted), candidate_value))
            attempts += 1
            if accepted:
                point = candidate
                value, gradient, hessian = candidate_value, candidate_gradient, candidate_hessian
                updated = True
                break
            step /= 2
    return point, leaf_weights, trace


def single_leaf(targets, **kwargs):
    targets = np.asarray(targets, np.float32)
    empty = np.array([], np.uint32)
    return _native.train(np.zeros((1, targets.size), np.uint8), targets, empty, empty,
                         **options(**kwargs))


@pytest.mark.parametrize("mode", MODES)
def test_zero_gradient_and_flat_huber_accept_equal_values(metal_device, mode):
    for objective, parameter, targets in (("RMSE", None, [0, 0]), ("Huber", 0, [-1, 7])):
        actual = single_leaf(targets, objective=objective, objective_param=parameter,
                             leaf_estimation_backtracking=mode, l2_leaf_reg=0)
        np.testing.assert_array_equal(actual.leaf_values, 0)
        np.testing.assert_array_equal(actual.loss, 0)
        assert actual.stats["kernel_dispatches"] > 0


@pytest.mark.parametrize("mode,expected", [("AnyImprovement", 0), ("Armijo", 100)])
def test_equal_loss_acceptance_distinguishes_armijo(metal_device, mode, expected):
    # Direction=200. Trial 200 has the SAME Huber loss as current 0.
    # AnyImprovement accepts it and then returns to 0 on its second attempt.
    # Armijo rejects it and spends its second attempt reaching the optimum 100.
    kwargs = dict(objective="Huber", objective_param=1, l2_leaf_reg=0.005,
                  leaf_estimation_iterations=2, leaf_estimation_backtracking=mode)
    actual = single_leaf([100], **kwargs)
    np.testing.assert_array_equal(actual.predictions, [expected])
    point, _, trace = cuda_leaf_walker([100], [0], [1], [0], 1,
                                       leaf_estimation_method="Newton", **kwargs)
    np.testing.assert_array_equal(point, [expected])
    assert len(trace) == 2
    assert [step[1] for step in trace] == ([True, True] if mode == "AnyImprovement" else [False, True])


@pytest.mark.parametrize("mode", MODES)
def test_first_acceptance_can_exceed_the_configured_iteration_budget(metal_device, mode):
    kwargs = dict(objective="Huber", objective_param=0.1, l2_leaf_reg=1e-6,
                  leaf_estimation_iterations=2, leaf_estimation_backtracking=mode)
    point, _, trace = cuda_leaf_walker([1], [0], [1], [0], 1,
                                       leaf_estimation_method="Newton", **kwargs)
    assert len(trace) == 17
    assert [item[1] for item in trace] == [False] * 16 + [True]
    np.testing.assert_array_equal(point, [np.float32(100000 / 65536)])
    actual = single_leaf([1], **kwargs)
    np.testing.assert_array_equal(actual.predictions, point)


@pytest.mark.parametrize("mode", MODES)
def test_nonfinite_poisson_trials_are_rejected_then_recover(metal_device, mode):
    kwargs = dict(objective="Poisson", l2_leaf_reg=0, leaf_estimation_iterations=2,
                  leaf_estimation_backtracking=mode)
    point, _, trace = cuda_leaf_walker([10], [-10], [1], [0], 1,
                                       leaf_estimation_method="Newton", **kwargs)
    assert len(trace) == 15 and sum(not np.isfinite(t[2]) for t in trace) > 0
    assert trace[-1][1]
    actual = single_leaf([10], bias=-10, **kwargs)
    np.testing.assert_allclose(actual.leaf_values[0], point, rtol=2e-6, atol=2e-6)
    assert actual.loss[-1] < 0
    assert np.isfinite(actual.predictions).all()


@pytest.mark.parametrize("mode", MODES)
def test_zero_weight_poisson_overflow_cannot_change_backtracking(metal_device, mode):
    kwargs = dict(objective="Poisson", l2_leaf_reg=0, leaf_estimation_iterations=2,
                  leaf_estimation_backtracking=mode)
    active = single_leaf([10], bias=-10, **kwargs)
    excluded = single_leaf([10, 1e30], sample_weight=np.array([1, 0], np.float32),
                           initial_predictions=np.array([-10, 1000], np.float32), **kwargs)
    np.testing.assert_array_equal(excluded.leaf_values, active.leaf_values)
    np.testing.assert_array_equal(excluded.loss, active.loss)
    np.testing.assert_array_equal(excluded.predictions[:1], active.predictions)
    assert np.isfinite(excluded.predictions).all()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_sample_weight_empty_leaf_remains_zero(metal_device, mode, method):
    candidate = np.array([0], np.uint32)
    result = _native.train(
        np.array([[0, 0, 1, 1]], np.uint8), np.array([-1, 1, 1e30, -1e30], np.float32),
        candidate, candidate,
        **options(depth=1, sample_weight=np.array([1, 2, 0, 0], np.float32),
                  leaf_estimation_method=method, leaf_estimation_backtracking=mode))
    np.testing.assert_array_equal(result.depths, [1])
    np.testing.assert_array_equal(result.leaf_weights, [[3, 0]])
    np.testing.assert_array_equal(result.leaf_values[:, 1], 0)
    np.testing.assert_array_equal(result.predictions[2:], 0)
    assert np.isfinite(result.loss).all()


def random_case(objective):
    rng = np.random.default_rng(658)
    bins = rng.integers(0, 6, size=(3, 769), dtype=np.uint8)
    signal = (1.7 * (bins[0] > 2) - 0.8 * (bins[1] > 3)
              + 0.05 * rng.normal(size=bins.shape[1]))
    if objective == "Poisson":
        targets = np.exp(signal / 2)
    elif objective in ("Logloss", "CrossEntropy"):
        targets = 1 / (1 + np.exp(-signal))
        if objective == "Logloss":
            targets = (rng.random(targets.size) < targets).astype(float)
    else:
        targets = signal
    weights = (0.25 + 2 * rng.random(targets.size)).astype(np.float32)
    weights[::17] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    return bins, targets.astype(np.float32), features, borders, weights


@pytest.mark.parametrize("objective,parameter", OBJECTIVES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("mode", MODES)
def test_weighted_multileaf_training_matches_independent_cuda_walker(
        metal_device, objective, parameter, method, mode):
    bins, targets, features, borders, weights = random_case(objective)
    kwargs = options(iterations=3, depth=2, learning_rate=0.2, l2_leaf_reg=2,
                     objective=objective, objective_param=parameter, sample_weight=weights,
                     leaf_estimation_method=method, leaf_estimation_backtracking=mode)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    baseline = np.zeros(targets.size, np.float32)
    assert np.all(actual.depths >= 1)
    # Replay the returned tree structures through independent CUDA leaf
    # optimization. Split-search scoring is covered by the existing oracle.
    for tree, depth in enumerate(actual.depths):
        leaf_ids = np.zeros(targets.size, np.int64)
        for level in range(depth):
            feature, border = actual.split_features[tree, level], actual.split_bins[tree, level]
            leaf_ids |= (bins[feature] > border).astype(np.int64) << level
        count = 1 << int(depth)
        point, leaf_weights, _ = cuda_leaf_walker(
            targets, baseline, weights, leaf_ids, count, objective=objective,
            objective_param=parameter, l2_leaf_reg=kwargs["l2_leaf_reg"],
            leaf_estimation_method=method, leaf_estimation_iterations=kwargs["leaf_estimation_iterations"],
            leaf_estimation_backtracking=mode)
        values = (point * np.float32(kwargs["learning_rate"])).astype(np.float32)
        np.testing.assert_allclose(actual.leaf_values[tree, :count], values, rtol=1e-4, atol=2e-5)
        np.testing.assert_allclose(actual.leaf_weights[tree, :count], leaf_weights, rtol=1e-6)
        baseline = (baseline + values[leaf_ids]).astype(np.float32)
        expected_loss = weighted_loss(targets, baseline, weights, objective, parameter)
        np.testing.assert_allclose(actual.loss[tree + 1], expected_loss, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(actual.predictions, baseline, rtol=1e-4, atol=3e-5)
    assert actual.stats["kernel_dispatches"] > 0


@pytest.mark.parametrize("objective,parameter", OBJECTIVES)
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
@pytest.mark.parametrize("mode", MODES)
def test_one_leaf_iteration_is_bitwise_identical_to_no_backtracking(
        metal_device, objective, parameter, method, mode):
    bins, targets, features, borders, weights = random_case(objective)
    kwargs = options(iterations=2, depth=2, objective=objective, objective_param=parameter,
                     sample_weight=weights, leaf_estimation_method=method, leaf_estimation_iterations=1)
    actual = _native.train(bins, targets, features, borders, **{**kwargs, "leaf_estimation_backtracking": mode})
    no = _native.train(bins, targets, features, borders, **{**kwargs, "leaf_estimation_backtracking": "No"})
    for field in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights", "predictions", "loss"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(no, field))
    assert actual.stats["kernel_dispatches"] == no.stats["kernel_dispatches"]


@pytest.mark.parametrize("objective,parameter", OBJECTIVES)
def test_existing_no_backtracking_path_matches_previous_independent_oracle(metal_device, objective, parameter):
    bins, targets, features, borders, weights = random_case(objective)
    kwargs = options(iterations=2, depth=2, learning_rate=0.2, objective=objective,
                     objective_param=parameter, sample_weight=weights, leaf_estimation_backtracking="No")
    expected = train_reference(bins, targets, features, borders, **kwargs)
    actual = _native.train(bins, targets, features, borders, **kwargs)
    np.testing.assert_array_equal(actual.depths, expected["depths"])
    np.testing.assert_array_equal(actual.split_features, expected["split_features"])
    np.testing.assert_array_equal(actual.split_bins, expected["split_bins"])
    for field in ("leaf_values", "predictions", "loss"):
        np.testing.assert_allclose(getattr(actual, field), expected[field], rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("objective,parameter", [("RMSE", None), ("Poisson", None), ("Huber", 0.8)])
def test_session_resume_from_saved_gpu_predictions_is_exact(metal_device, tmp_path, mode, objective, parameter):
    bins, targets, features, borders, weights = random_case(objective)
    kwargs = options(iterations=5, depth=2, learning_rate=0.2, objective=objective,
                     objective_param=parameter, sample_weight=weights, leaf_estimation_backtracking=mode)
    uninterrupted = _native.train(bins, targets, features, borders, **kwargs)
    with _native.Session(bins, targets, features, borders, **kwargs) as session:
        for _ in range(2):
            session.step()
        first = session.result()
        np.testing.assert_array_equal(session.predictions(), first.predictions)
        state = tmp_path / "cursor.npy"
        np.save(state, session.predictions())
    with _native.Session(bins, targets, features, borders,
                         **{**kwargs, "iterations": 3, "initial_predictions": np.load(state),
                            "iteration_offset": 2}) as resumed:
        for _ in range(3):
            resumed.step()
        rest = resumed.result()
    for field in ("depths", "split_features", "split_bins", "leaf_values", "leaf_weights"):
        combined = np.concatenate([getattr(first, field), getattr(rest, field)])
        np.testing.assert_array_equal(combined, getattr(uninterrupted, field))
    np.testing.assert_array_equal(rest.predictions, uninterrupted.predictions)
    np.testing.assert_array_equal(np.concatenate([first.loss, rest.loss[1:]]), uninterrupted.loss)


@pytest.mark.parametrize("mode", [None, "", "armijo", "Armijo ", "Invalid", 3, True])
def test_invalid_backtracking_is_rejected_before_loading_the_device(monkeypatch, mode):
    def forbidden():
        pytest.fail("invalid backtracking must fail before loading the Metal library")
    monkeypatch.setattr(_native, "build_library", forbidden)
    with pytest.raises(ValueError, match="leaf_estimation_backtracking"):
        single_leaf([1], leaf_estimation_backtracking=mode)
