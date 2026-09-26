"""Independent structure/leaf checks for persistent GPU permutation cursors.

The selected permutation supplies structure gradients and bin routing. Every
permutation then fits that shared structure using its own bins and predictions;
only the last permutation supplies the exported tree and public predictions.
"""

import platform

import numpy as np
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from catboost_metal import _native
from cuda_reference import _score_children
from cuda_scalar_reference import objective_terms, weighted_loss
from test_backtracking import cuda_leaf_walker


@pytest.fixture(autouse=True)
def prohibit_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Permutation session tests must not train CPU CatBoost")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)
    monkeypatch.setattr(CatBoostClassifier, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon")
    return _native.device_info()


def problem(count, objective="RMSE", rows=259):
    rng = np.random.default_rng(937)
    original = rng.integers(0, 6, (3, rows), dtype=np.uint8)
    matrices = [original.copy()]
    for permutation in range(1, count):
        matrix = original.copy()
        for feature in range(matrix.shape[0]):
            rng.shuffle(matrix[feature])
        matrices.append(matrix)
    signal = 1.3 * (original[0] > 2) - 0.7 * (original[1] > 3) + rng.normal(0, 0.15, rows)
    if objective == "Poisson":
        targets = np.exp(signal / 2)
    elif objective in ("Logloss", "CrossEntropy"):
        targets = 1 / (1 + np.exp(-signal))
        if objective == "Logloss":
            targets = (rng.random(rows) < targets).astype(float)
    else:
        targets = signal
    weights = rng.uniform(0.25, 2, rows).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    initial = rng.normal(0, 0.1, (count, rows)).astype(np.float32)
    return original, matrices, targets.astype(np.float32), weights, features, borders, initial


def options(**overrides):
    result = dict(iterations=4, depth=2, learning_rate=0.2, l2_leaf_reg=2, bias=0,
                  score_function="Cosine", objective="RMSE", objective_param=None,
                  leaf_estimation_iterations=3, leaf_estimation_method="Newton",
                  leaf_estimation_backtracking="No", random_seed=671)
    result.update(overrides)
    return result


def structure_reference(bins, targets, prediction, weights, features, borders, params):
    """Use masked scalar CUDA scores, independent of the histogram backend."""
    _, unweighted, _ = objective_terms(targets, prediction, params["objective"], params["objective_param"])
    gradients = (weights.astype(np.float64) * unweighted).astype(np.float32).astype(np.float64)
    regularization = float(np.float32(params["l2_leaf_reg"])) or float(np.float32(1e-20))
    leaf_ids = np.zeros(targets.size, np.int64)
    splits = []
    for level in range(params["depth"]):
        scores = []
        for feature, border in zip(features, borders):
            sums, child_weights = [], []
            for leaf in range(1 << level):
                member = leaf_ids == leaf
                left = member & (bins[feature] <= border)
                right = member & ~left
                sums.extend([gradients[left].sum(), gradients[right].sum()])
                child_weights.extend([weights[left].sum(dtype=np.float64), weights[right].sum(dtype=np.float64)])
            scores.append(_score_children(np.asarray(sums), np.asarray(child_weights), regularization,
                                           params["score_function"]))
        if not scores:
            break
        winner = int(np.argmin(scores))
        split = (int(features[winner]), int(borders[winner]))
        if split in splits:
            break
        splits.append(split)
        leaf_ids |= (bins[split[0]] > split[1]).astype(np.int64) << level
    return splits


def fit_fixed_structure(matrix, targets, cursor, weights, splits, params):
    leaf_ids = np.zeros(targets.size, np.int64)
    for level, (feature, border) in enumerate(splits):
        leaf_ids |= (matrix[feature] > border).astype(np.int64) << level
    point, leaf_weights, _ = cuda_leaf_walker(
        targets, cursor, weights, leaf_ids, 1 << len(splits),
        objective=params["objective"], objective_param=params["objective_param"],
        l2_leaf_reg=params["l2_leaf_reg"], leaf_estimation_method=params["leaf_estimation_method"],
        leaf_estimation_iterations=params["leaf_estimation_iterations"],
        leaf_estimation_backtracking=params["leaf_estimation_backtracking"])
    values = (point * np.float32(params["learning_rate"])).astype(np.float32)
    prediction = (cursor + values[leaf_ids]).astype(np.float32)
    return prediction, values, leaf_weights


def assert_result_equal(left, right):
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values",
                 "leaf_weights", "predictions", "loss"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name), err_msg=name)


@pytest.mark.parametrize("bootstrap", [{}, {"bootstrap_type": "MVS", "subsample": 0.6}])
def test_one_permutation_matches_the_existing_session_exactly(metal_device, bootstrap):
    original, matrices, targets, weights, features, borders, initial = problem(1)
    params = options(sample_weight=weights, initial_predictions=initial[0], **bootstrap)
    expected = _native.train(original, targets, features, borders, **params)
    with _native.Session(original, targets, features, borders, **params) as session:
        session.configure_permutations(matrices)
        np.testing.assert_array_equal(session.permutation_state["predictions"], initial)
        for _ in range(params["iterations"]):
            session.select_permutation(0)
            session.step()
        assert_result_equal(session.result(), expected)


@pytest.mark.parametrize("count", [1, 2, 3, 4])
@pytest.mark.parametrize("score", ["L2", "Cosine"])
@pytest.mark.parametrize("objective,parameter,method,mode", [
    ("RMSE", None, "Newton", "No"),
    ("CrossEntropy", None, "Newton", "Armijo"),
    ("Huber", 0.8, "Gradient", "AnyImprovement"),
])
def test_chosen_structure_and_every_fixed_leaf_estimation_match_oracles(
        metal_device, count, score, objective, parameter, method, mode):
    original, matrices, targets, weights, features, borders, initial = problem(count, objective)
    params = options(objective=objective, objective_param=parameter, score_function=score,
                     leaf_estimation_method=method, leaf_estimation_backtracking=mode)
    expected_cursors = initial.copy()
    with _native.Session(original, targets, features, borders, **params, sample_weight=weights) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        state = session.permutation_state
        assert state["predictions"].shape == initial.shape
        assert state["predictions"].dtype == np.float32
        assert state["mvs_lambdas"].shape == (count,)
        assert state["mvs_valid"].shape == (count,)
        np.testing.assert_array_equal(state["predictions"], initial)
        np.testing.assert_allclose(session.result().loss[0],
                                   weighted_loss(targets, initial[-1], weights, objective, parameter), rtol=2e-6)
        for iteration in range(params["iterations"]):
            chosen = (count - 1 - iteration) % count
            expected_splits = structure_reference(matrices[chosen], targets, expected_cursors[chosen],
                                                   weights, features, borders, params)
            session.select_permutation(chosen)
            step = session.step()
            actual_splits = list(zip(step.split_features.tolist(), step.split_bins.tolist()))
            assert actual_splits == expected_splits
            expected_values = expected_weights = None
            for permutation, matrix in enumerate(matrices):
                expected_cursors[permutation], values, leaf_weights = fit_fixed_structure(
                    matrix, targets, expected_cursors[permutation], weights, expected_splits, params)
                if permutation == count - 1:
                    expected_values, expected_weights = values, leaf_weights
            state = session.permutation_state
            np.testing.assert_allclose(state["predictions"], expected_cursors, rtol=2e-4, atol=5e-5)
            np.testing.assert_allclose(step.leaf_values, expected_values, rtol=2e-4, atol=5e-5)
            np.testing.assert_allclose(step.leaf_weights, expected_weights, rtol=2e-6)
            np.testing.assert_array_equal(session.predictions(), state["predictions"][-1])
            np.testing.assert_array_equal(session.result().predictions, state["predictions"][-1])
            np.testing.assert_allclose(step.loss, weighted_loss(targets, state["predictions"][-1], weights,
                                                               objective, parameter), rtol=3e-6, atol=2e-6)


def test_configure_replaces_the_original_permutation_zero_matrix(metal_device):
    original, matrices, targets, weights, features, borders, _ = problem(2)
    replacement = matrices[1]
    expected = _native.train(replacement, targets, features, borders, **options(sample_weight=weights))
    with _native.Session(original, targets, features, borders, **options(sample_weight=weights)) as session:
        session.configure_permutations([replacement])
        for _ in range(4):
            session.select_permutation(0)
            session.step()
        assert_result_equal(session.result(), expected)


def test_omitted_cursor_and_mvs_state_clone_the_initial_session_state(metal_device):
    original, matrices, targets, weights, features, borders, initial = problem(4)
    params = options(sample_weight=weights, initial_predictions=initial[-1],
                     bootstrap_type="MVS", subsample=0.6, initial_mvs_lambda=0.75)
    with _native.Session(original, targets, features, borders, **params) as session:
        session.configure_permutations(matrices)
        state = session.permutation_state
        np.testing.assert_array_equal(state["predictions"], np.repeat(initial[-1:], 4, axis=0))
        np.testing.assert_array_equal(state["mvs_lambdas"], np.full(4, 0.75, np.float32))
        np.testing.assert_array_equal(state["mvs_valid"], np.ones(4, np.uint8))


@pytest.mark.parametrize("mode", ["No", "Armijo"])
def test_mvs_uses_each_permutations_previous_shrunk_leaf_values(metal_device, mode):
    original, matrices, targets, weights, features, borders, initial = problem(4)
    params = options(bootstrap_type="MVS", subsample=0.6, leaf_estimation_backtracking=mode)
    previous = initial.copy()
    with _native.Session(original, targets, features, borders, **params, sample_weight=weights) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        for chosen in (2, 0, 3, 1):
            session.select_permutation(chosen)
            step = session.step()
            splits = list(zip(step.split_features.tolist(), step.split_bins.tolist()))
            expected_lambdas = []
            for permutation, matrix in enumerate(matrices):
                previous[permutation], values, leaf_weights = fit_fixed_structure(
                    matrix, targets, previous[permutation], weights, splits, params)
                expected_lambdas.append(np.float32(np.mean(np.abs(values), dtype=np.float64) ** 2))
                if permutation == len(matrices) - 1:
                    np.testing.assert_allclose(step.leaf_weights, leaf_weights, rtol=2e-6)
            state = session.permutation_state
            np.testing.assert_array_equal(state["mvs_valid"], np.ones(4, np.uint8))
            np.testing.assert_allclose(state["mvs_lambdas"], expected_lambdas, rtol=2e-4, atol=1e-8)
            np.testing.assert_allclose(state["predictions"], previous, rtol=2e-4, atol=5e-5)
        assert np.ptp(state["mvs_lambdas"]) > 1e-6


def run_session(data, params, choices, initial=None, lambdas=None, valid=None):
    original, matrices, targets, weights, features, borders, default_initial = data
    with _native.Session(original, targets, features, borders, **params, sample_weight=weights) as session:
        session.configure_permutations(matrices, initial_predictions=default_initial if initial is None else initial,
                                       mvs_lambdas=lambdas, mvs_valid=valid)
        for chosen in choices:
            session.select_permutation(chosen)
            session.step()
        return session.result(), session.permutation_state


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"},
    {"bootstrap_type": "Bayesian", "bagging_temperature": 1.5},
    {"bootstrap_type": "Bernoulli", "subsample": 0.5},
    {"bootstrap_type": "Poisson", "subsample": 0.6},
    {"bootstrap_type": "MVS", "subsample": 0.6},
])
def test_all_cursor_and_mvs_snapshots_resume_noisy_training_exactly(metal_device, tmp_path, bootstrap):
    data = problem(4)
    params = options(iterations=6, random_strength=1.2, **bootstrap)
    choices = [2, 0, 3, 1, 2, 0]
    full, full_state = run_session(data, params, choices)
    repeated, repeated_state = run_session(data, params, choices)
    assert_result_equal(repeated, full)
    for name in ("predictions", "mvs_lambdas", "mvs_valid"):
        np.testing.assert_array_equal(repeated_state[name], full_state[name])
    first, state = run_session(data, params, choices[:3])
    path = tmp_path / "permutations.npz"
    np.savez(path, **state)
    with np.load(path, allow_pickle=False) as saved:
        restored = {key: saved[key].copy() for key in ("predictions", "mvs_lambdas", "mvs_valid")}
    resumed, resumed_state = run_session(
        data, {**params, "iterations": 3, "iteration_offset": 3}, choices[3:],
        initial=restored["predictions"], lambdas=restored["mvs_lambdas"], valid=restored["mvs_valid"])
    for name in ("depths", "split_features", "split_bins", "split_types", "leaf_values", "leaf_weights"):
        np.testing.assert_array_equal(np.concatenate([getattr(first, name), getattr(resumed, name)]),
                                      getattr(full, name), err_msg=name)
    np.testing.assert_array_equal(np.concatenate([first.loss, resumed.loss[1:]]), full.loss)
    np.testing.assert_array_equal(resumed.predictions, full.predictions)
    for name in ("predictions", "mvs_lambdas", "mvs_valid"):
        np.testing.assert_array_equal(resumed_state[name], full_state[name], err_msg=name)


@pytest.mark.parametrize("kind", [
    "empty", "too_many", "wrong_rows", "wrong_features", "float_bins", "negative_bins", "larger_bins",
    "wrong_predictions", "nan_predictions", "wrong_lambdas", "negative_lambdas", "nan_lambdas",
    "invalid_flags", "flags_without_lambdas", "lambdas_without_flags",
])
def test_invalid_configuration_fails_before_native_mutation(metal_device, monkeypatch, kind):
    original, matrices, targets, weights, features, borders, _ = problem(2, rows=17)
    kwargs = {}
    if kind == "empty": matrices = []
    elif kind == "too_many": matrices = [original] * 65
    elif kind == "wrong_rows": matrices[1] = matrices[1][:, :-1]
    elif kind == "wrong_features": matrices[1] = matrices[1][:-1]
    elif kind == "float_bins": matrices[1] = matrices[1].astype(np.float32)
    elif kind == "negative_bins": matrices[1] = np.full_like(matrices[1], -1, dtype=np.int32)
    elif kind == "larger_bins": matrices[1] = np.full_like(matrices[1], 6)
    elif kind == "wrong_predictions": kwargs["initial_predictions"] = np.zeros((2, 16), np.float32)
    elif kind == "nan_predictions": kwargs["initial_predictions"] = np.full((2, 17), np.nan, np.float32)
    elif kind == "wrong_lambdas": kwargs.update(mvs_lambdas=[0], mvs_valid=[1, 1])
    elif kind == "negative_lambdas": kwargs.update(mvs_lambdas=[-1, 0], mvs_valid=[1, 1])
    elif kind == "nan_lambdas": kwargs.update(mvs_lambdas=[np.nan, 0], mvs_valid=[1, 1])
    elif kind == "invalid_flags": kwargs.update(mvs_lambdas=[0, 0], mvs_valid=[0, 2])
    elif kind == "flags_without_lambdas": kwargs["mvs_valid"] = [1, 1]
    elif kind == "lambdas_without_flags": kwargs["mvs_lambdas"] = [0, 0]

    class ForbiddenNativeCalls:
        def __getattr__(self, name):
            pytest.fail(f"Invalid permutation configuration reached native call {name}")

    with _native.Session(original, targets, features, borders, **options(sample_weight=weights)) as session:
        with monkeypatch.context() as patch:
            patch.setattr(session, "_lib", ForbiddenNativeCalls())
            with pytest.raises((ValueError, TypeError)):
                session.configure_permutations(matrices, **kwargs)


@pytest.mark.parametrize("index", [-1, 2, 64, True, 0.5, "0"])
def test_invalid_selection_does_not_change_cursors(metal_device, index):
    original, matrices, targets, weights, features, borders, initial = problem(2)
    with _native.Session(original, targets, features, borders, **options(sample_weight=weights)) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        with pytest.raises((ValueError, TypeError)):
            session.select_permutation(index)
        np.testing.assert_array_equal(session.permutation_state["predictions"], initial)


def test_configuration_cannot_discard_an_existing_training_step(metal_device):
    original, matrices, targets, weights, features, borders, _ = problem(2)
    with _native.Session(original, targets, features, borders, **options(sample_weight=weights)) as session:
        session.step()
        before = session.predictions()
        with pytest.raises((ValueError, RuntimeError)):
            session.configure_permutations(matrices)
        np.testing.assert_array_equal(session.predictions(), before)


def test_configuration_can_only_be_installed_once(metal_device):
    original, matrices, targets, weights, features, borders, initial = problem(2)
    with _native.Session(original, targets, features, borders, **options(sample_weight=weights)) as session:
        session.configure_permutations(matrices, initial_predictions=initial)
        with pytest.raises((ValueError, RuntimeError)):
            session.configure_permutations([original])
        np.testing.assert_array_equal(session.permutation_state["predictions"], initial)
