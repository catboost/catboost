"""Independent scalar equations, actual Metal training, and standard model readers."""
import ctypes as ct
import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal import _greedy
from catboost_metal._greedy_model import model_json
from catboost_metal._greedy_training import run_training
from test_greedy_training import route


# Native IDs intentionally match the shared scalar ABI. Lq is a Metal extension:
# upstream CUDA has its derivatives but does not register greedy Lq trainers.
CASES = (
    ("RMSE", None, "Newton"), ("Logloss", None, "Newton"),
    ("CrossEntropy", None, "Newton"), ("Poisson", None, "Newton"),
    ("Huber", .7, "Newton"), ("Expectile", .73, "Newton"),
    ("Lq", 2.7, "Newton"), ("Tweedie", 1.4, "Newton"),
    ("LogLinQuantile", .31, "Gradient"), ("Quantile", .31, "Gradient"),
    ("MAE", None, "Gradient"), ("MAPE", None, "Gradient"),
)
SCORES = ("L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2")


@pytest.fixture(autouse=True)
def actual_metal_without_cpu_fitting(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs):
        raise AssertionError("These tests must not fit CPU CatBoost models")
    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


def problem(objective, rows=263):
    rng = np.random.default_rng(3051)
    bins = rng.integers(0, 7, (3, rows), dtype=np.uint8)
    signal = .8 * (bins[0] > 2) - .65 * (bins[1] > 3) + .35 * (bins[2] > 1)
    y = signal + rng.uniform(-.23, .23, rows)
    if objective in ("Poisson", "Tweedie", "LogLinQuantile"):
        y = .2 + 1.8 * np.exp(y)
    elif objective == "Logloss":
        y = rng.random(rows) < 1 / (1 + np.exp(-2 * signal))
    elif objective == "CrossEntropy":
        y = 1 / (1 + np.exp(-2 * y))
    w = rng.uniform(.3, 2.4, rows).astype(np.float32)
    w[::17] = 0
    return (bins, np.asarray(y, np.float32), w,
            np.repeat(np.arange(3, dtype=np.uint32), 6), np.tile(np.arange(6, dtype=np.uint32), 3))


def objective_equations(raw, y, objective, parameter):
    """Unweighted loss, negative derivative, curvature without port helpers."""
    raw, y = np.asarray(raw, float), np.asarray(y, float)
    residual = y - raw
    if objective == "RMSE":
        return residual**2, residual, np.ones_like(raw)
    if objective in ("Logloss", "CrossEntropy"):
        probability = 1 / (1 + np.exp(-raw))
        return np.logaddexp(0, raw) - y * raw, y - probability, probability * (1 - probability)
    if objective == "Poisson":
        return np.exp(raw) - y * raw, y - np.exp(raw), np.exp(raw)
    if objective == "Huber":
        inside = np.abs(residual) < parameter
        return (np.where(inside, .5 * residual**2, parameter * (np.abs(residual) - .5 * parameter)),
                np.clip(residual, -parameter, parameter), inside.astype(float))
    if objective == "Expectile":
        scale = np.where(residual > 0, parameter, 1 - parameter)
        return scale * residual**2, 2 * scale * residual, 2 * scale
    if objective == "Lq":
        magnitude = np.abs(residual)
        return (magnitude**parameter,
                parameter * np.where(residual > 0, 1, -1) * magnitude**(parameter - 1),
                parameter * (parameter - 1) * magnitude**(parameter - 2)
                if parameter >= 2 else np.ones_like(raw))
    if objective == "Tweedie":
        first = y * np.exp((1 - parameter) * raw)
        second = np.exp((2 - parameter) * raw)
        return (-first / (1 - parameter) + second / (2 - parameter), first - second,
                (parameter - 1) * first + (2 - parameter) * second)
    if objective in ("Quantile", "MAE", "LogLinQuantile"):
        alpha = .5 if objective == "MAE" else parameter
        prediction = np.exp(raw) if objective == "LogLinQuantile" else raw
        residual = y - prediction
        multiplier = np.where(residual > 0, alpha, -(1 - alpha))
        return (multiplier * residual, multiplier * prediction if objective == "LogLinQuantile" else multiplier,
                np.zeros_like(raw))
    denominator = np.maximum(1, np.abs(y))
    return np.abs(residual) / denominator, np.where(residual > 0, 1, -1) / denominator, np.zeros_like(raw)


def expected_metric(raw, y, w, objective, parameter):
    loss = objective_equations(raw, y, objective, parameter)[0]
    result = np.average(loss, weights=w)
    return np.sqrt(result) if objective == "RMSE" else result * (2 if objective == "MAE" else 1)


def expected_root(bins, y, w, cf, cb, raw, objective, parameter, score, l2):
    _, gradient, hessian = objective_equations(raw, y, objective, parameter)
    gradient *= w
    denominator = w * hessian if score in ("NewtonL2", "NewtonCosine") else w

    def score_parts(sums, weights):
        if score == "SolarL2":
            return sum(-g*g*(1 + 2*np.log1p(v))/v for g, v in zip(sums, weights) if v > 1e-20)
        if score == "LOOL2":
            return sum(-g*g/v*(v/(v-1))**2 for g, v in zip(sums, weights) if v > 1)
        if score in ("L2", "NewtonL2"):
            return sum(-g*g/(v+l2) for g, v in zip(sums, weights) if v > 1e-20)
        directions = [g/(v+l2) if v > 0 else 0 for g, v in zip(sums, weights)]
        return -sum(g*d for g, d in zip(sums, directions))/np.sqrt(
            1e-10 + sum(v*d*d for v, d in zip(weights, directions)))

    before = score_parts([gradient.sum()], [denominator.sum()])
    gains = []
    for f, b in zip(cf, cb):
        left = bins[f] <= b
        weights = [denominator[left].sum(), denominator[~left].sum()]
        gains.append(0 if min(weights) < 1e-20 else
            score_parts([gradient[left].sum(), gradient[~left].sum()], weights) - before)
    winner = int(np.argmin(gains))
    return winner, gains[winner]


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
@pytest.mark.parametrize("score", SCORES)
def test_all_objectives_scores_weighted_root_and_iterated_leaves(case, score):
    objective, parameter, method = case
    bins, y, w, cf, cb = problem(objective)
    policy = ("Depthwise", "Lossguide", "Region")[SCORES.index(score) % 3]
    result = _greedy.train(bins, y, cf, cb, sample_weight=w, objective=objective,
        objective_param=parameter, leaf_estimation_method=method, leaf_estimation_iterations=3,
        score_function=score, grow_policy=policy, iterations=2, depth=3, max_leaves=6,
        bias=.13, learning_rate=.18, l2_leaf_reg=2.3)
    raw = np.full(len(y), np.float32(.13), np.float32)
    winner, gain = expected_root(bins, y, w.astype(float), cf, cb, raw, objective, parameter, score, 2.3)
    first = result.trees[0]
    if policy != "Depthwise" or gain < 0:
        assert first.nodes[0, :3].tolist() == [int(cf[winner]), int(cb[winner]), 0]
    else:
        assert len(first.nodes) == 1
    assert result.loss[0] == pytest.approx(expected_metric(raw, y, w, objective, parameter), rel=3e-6, abs=2e-7)
    for tree in result.trees:
        ids = route(tree, bins)
        expected = np.zeros(len(tree.leaf_values), np.float32)
        totals = np.bincount(ids, weights=w, minlength=len(expected))
        for _ in range(3):
            _, gradient, hessian = objective_equations(raw + expected[ids], y, objective, parameter)
            for leaf in range(len(expected)):
                selected = ids == leaf
                diagonal = np.sum(w[selected] * hessian[selected]) if method == "Newton" else totals[leaf]
                if totals[leaf] < 1e-20:
                    expected[leaf] = 0
                elif diagonal + 2.3 > 0:
                    expected[leaf] += np.sum(w[selected] * gradient[selected])/(diagonal + 2.3 + 1e-20)
        np.testing.assert_allclose(tree.leaf_values, expected*.18, rtol=7e-5, atol=3e-6)
        np.testing.assert_allclose(tree.leaf_weights, totals, rtol=3e-6, atol=2e-6)
        raw += tree.leaf_values[ids]
        assert tree.loss == pytest.approx(expected_metric(raw, y, w, objective, parameter), rel=3e-5, abs=2e-6)
    np.testing.assert_array_equal(result.predictions, raw)
    assert result.stats["kernel_dispatches"] > 0 and result.stats["device"]


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_scalar_model_readers_and_parameter_metadata(case, tmp_path):
    objective, parameter, method = case
    bins, y, w, cf, cb = problem(objective)
    result = _greedy.train(bins, y, cf, cb, objective=objective, objective_param=parameter,
        sample_weight=w, leaf_estimation_method=method, score_function="SolarL2", iterations=2, depth=2,
        bias=.13, learning_rate=.18)
    model = model_json(result, [np.arange(6) + .5]*3, objective=objective, objective_param=parameter, bias=.13)
    key = _greedy.OBJECTIVE_PARAMETERS.get(objective)
    loss_metadata = model["model_info"]["params"]["loss_function"]
    assert loss_metadata["type"] == objective
    if key:
        assert float(loss_metadata["params"][key]) == pytest.approx(parameter)
    if objective == "Lq":
        assert "does not register Lq" in model["model_info"]["metal_objective_scope"]
    source = tmp_path / "model.json"
    source.write_text(json.dumps(model, allow_nan=False))
    reader = CatBoost().load_model(source, format="json")
    np.testing.assert_allclose(reader.predict(bins.T, prediction_type="RawFormulaVal"),
                               result.predictions, rtol=2e-5, atol=2e-6)
    reader.save_model(tmp_path / "model.cbm")
    restored = CatBoost().load_model(tmp_path / "model.cbm")
    np.testing.assert_allclose(restored.predict(bins.T, prediction_type="RawFormulaVal"),
                               result.predictions, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("case", CASES, ids=[case[0] for case in CASES])
def test_parameterized_validation_snapshot_and_resume(case, tmp_path):
    objective, parameter, method = case
    bins, y, w, cf, cb = problem(objective)
    options = dict(iterations=4, depth=2, learning_rate=.12, l2_leaf_reg=2.3, bias=.13,
        score_function="LOOL2", objective=objective, objective_param=parameter,
        leaf_estimation_method=method, sample_weight=w, eval_bins=bins[:, ::2],
        eval_targets=y[::2], eval_weight=w[::2], use_best_model=False)
    expected = run_training(bins, y, cf, cb, **options)
    path = tmp_path / "objective.npz"
    partial = run_training(bins, y, cf, cb, **options, save_snapshot=True, snapshot_file=path,
        callback=lambda info: info.iteration < 2)
    assert partial.completed_iterations == 2
    resumed = run_training(bins, y, cf, cb, **options, save_snapshot=True, snapshot_file=path)
    np.testing.assert_array_equal(resumed.predictions, expected.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions, expected.eval_predictions)
    np.testing.assert_array_equal(resumed.loss, expected.loss)
    metric_name = objective if parameter is None else f"{objective}:{_greedy.OBJECTIVE_PARAMETERS[objective]}={parameter!r}"
    assert set(resumed.evals_result["validation"]) == {metric_name}
    assert resumed.evals_result == expected.evals_result
    assert resumed.evals_result["validation"][metric_name][-1] == pytest.approx(
        expected_metric(resumed.eval_predictions, y[::2], w[::2], objective, parameter), rel=3e-5, abs=1e-6)
    if parameter is not None:
        with pytest.raises(ValueError, match="does not match"):
            changed = dict(options, objective_param=parameter + .01)
            run_training(bins, y, cf, cb, **changed, save_snapshot=True, snapshot_file=path)


def test_poisson_negative_metric_survives_snapshot_roundtrip(tmp_path):
    bins = np.zeros((1, 7), np.uint8)
    y, weights = np.full(7, 10, np.float32), np.arange(1, 8, dtype=np.float32)
    options = dict(iterations=3, depth=0, learning_rate=.2, l2_leaf_reg=2., bias=float(np.log(10)),
                   score_function="L2", objective="Poisson", sample_weight=weights,
                   save_snapshot=True, snapshot_file=tmp_path / "negative.npz")
    run_training(bins, y, [], [], **options, callback=lambda info: info.iteration < 1)
    result = run_training(bins, y, [], [], **options)
    assert (result.loss < 0).all()
    assert result.resumed_iterations == 1
    assert result.loss[-1] == pytest.approx(10 - 10*np.log(10), rel=2e-6)


@pytest.mark.parametrize("objective,parameter", [("Quantile", .31), ("MAE", None), ("MAPE", None)])
@pytest.mark.parametrize("backtracking", ["No", "Armijo"])
def test_exact_lifecycle_resume_and_standard_reader(objective, parameter, backtracking, tmp_path):
    bins, y, w, cf, cb = problem(objective)
    options = dict(iterations=4, depth=3, learning_rate=.2, l2_leaf_reg=2.7, bias=.13,
        score_function="SolarL2", objective=objective, objective_param=parameter,
        leaf_estimation_method="Exact", leaf_estimation_iterations=17,
        leaf_estimation_backtracking=backtracking, sample_weight=w, eval_bins=bins[:, ::2],
        eval_targets=y[::2], eval_weight=w[::2], use_best_model=False)
    expected = run_training(bins, y, cf, cb, **options)
    path = tmp_path / "exact.npz"
    partial = run_training(bins, y, cf, cb, **options, save_snapshot=True, snapshot_file=path,
        callback=lambda info: info.iteration < 2)
    assert partial.completed_iterations == 2
    resumed = run_training(bins, y, cf, cb, **options, save_snapshot=True, snapshot_file=path)
    np.testing.assert_array_equal(resumed.predictions, expected.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions, expected.eval_predictions)
    np.testing.assert_array_equal(resumed.loss, expected.loss)
    # SolarL2 ignores lambda during structure search. Exact values independently
    # ignore lambda, iterative step count, and backtracking acceptance checks.
    alternative = dict(options, l2_leaf_reg=.001, leaf_estimation_iterations=1,
                       leaf_estimation_backtracking="AnyImprovement")
    once = run_training(bins, y, cf, cb, **alternative)
    np.testing.assert_array_equal(once.predictions, expected.predictions)
    for a, b in zip(once.trees, expected.trees):
        np.testing.assert_array_equal(a.nodes, b.nodes)
        np.testing.assert_array_equal(a.leaf_values, b.leaf_values)
    model = model_json(resumed, [np.arange(6) + .5]*3, objective=objective,
                       objective_param=parameter, bias=.13)
    source = tmp_path / "exact.json"
    source.write_text(json.dumps(model, allow_nan=False))
    reader = CatBoost().load_model(source, format="json")
    np.testing.assert_allclose(reader.predict(bins.T, prediction_type="RawFormulaVal"),
                               resumed.predictions, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("options", [
    {"objective": "Huber"}, {"objective": "Tweedie", "objective_param": 2.},
    {"objective": "Expectile", "objective_param": 1.1}, {"objective": "Lq", "objective_param": .9},
    {"objective": "Huber", "objective_param": -1}, {"objective": "Quantile", "objective_param": 1.1},
    {"objective": "Lq", "objective_param": 1.5}, {"objective": "MAE"},
    {"objective": "Poisson", "leaf_estimation_method": "Exact"},
    {"objective": "CrossEntropy", "objective_param": float("nan")},
])
def test_objective_validation_before_native_call(options, monkeypatch):
    def forbidden():
        raise AssertionError("Invalid options must fail before loading Metal")
    monkeypatch.setattr(_greedy, "build_library", forbidden)
    with pytest.raises(ValueError):
        _greedy.TrainingSession(np.zeros((1, 2), np.uint8), [0, 1], [0], [0], **options)


@pytest.mark.parametrize("objective,targets", [("CrossEntropy", [0, 1.2]), ("Poisson", [-1, 1]), ("Tweedie", [-1, 1])])
def test_objective_target_domains(objective, targets):
    with pytest.raises(ValueError, match="targets"):
        _greedy.TrainingSession(np.zeros((1, 2), np.uint8), targets, [0], [0],
            objective=objective, objective_param=1.4 if objective == "Tweedie" else None)


def test_configured_native_constructor_checks_shared_objective_abi():
    bins, y, w, cf, cb = problem("Huber", 7)
    lib = _greedy._load(_greedy.build_library())
    p = _greedy.Params(7, 3, len(cf), 7, 1, 2, 3, 1, 1, 4, 0, 0, 1,
                       0, 0, 0, .2, 3., 0., 0)
    for objective in (_greedy.ObjectiveOptions(5, 0, .7, 0), _greedy.ObjectiveOptions(4, 1, .7, 0),
                      _greedy.ObjectiveOptions(4, 0, .7, 1), _greedy.ObjectiveOptions(4, 0, -1, 0)):
        handle, error = ct.c_void_p(), ct.create_string_buffer(2048)
        code = lib.cbm_greedy_session_create_configured(ct.byref(p), ct.byref(objective), _greedy._u8(bins),
            _greedy._f32(y), _greedy._f32(w), None, _greedy._u32(cf), _greedy._u32(cb), None,
            ct.byref(handle), error, len(error))
        try:
            assert code and error.value and not handle.value
        finally:
            lib.cbm_greedy_session_close(handle)
