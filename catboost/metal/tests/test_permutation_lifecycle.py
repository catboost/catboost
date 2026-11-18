"""Controller permutation scheduling and safe continuation on the Metal GPU.

The comparison drives a separate native Session directly. It uses the literal
CUDA permutation chooser, never CatBoost CPU training, and checks the exported
estimation permutation independently from each search permutation's state.
"""

import copy
import json
import platform

import numpy as np
import pytest

from catboost_metal import _native
from catboost_metal._data import cuda_search_permutation
from catboost_metal._training import metric, run_training


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Permutation lifecycle tests must not train CatBoost on CPU")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Permutation lifecycle checks require Apple Silicon")
    assert _native.device_info()["backend"] == "Metal"


def _problem(objective="RMSE", rows=259):
    rng = np.random.default_rng(9872)
    bins = rng.integers(0, 6, size=(3, rows), dtype=np.uint8)
    permutations = [bins.copy()]
    for _ in range(3):
        current = bins.copy()
        for column in current:
            rng.shuffle(column)
        permutations.append(current)
    signal = (1.3 * (bins[0] > 2) - .8 * (bins[1] > 3)
              + rng.normal(0., .1, rows)).astype(np.float32)
    targets = signal if objective == "RMSE" else (1 / (1 + np.exp(-signal))).astype(np.float32)
    weights = rng.uniform(.2, 2., rows).astype(np.float32)
    weights[::17] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    return bins, targets, features, borders, tuple(permutations), weights


def _options(**overrides):
    result = dict(iterations=9, depth=2, learning_rate=.2, l2_leaf_reg=2.,
                  bias=.03125, score_function="Cosine", objective="RMSE",
                  leaf_estimation_iterations=3, random_seed=671,
                  iteration_offset=17, random_strength=.7)
    result.update(overrides)
    return result


def _manual(data, options):
    bins, targets, features, borders, permutations, weights = data
    with _native.Session(bins, targets, features, borders, sample_weight=weights, **options) as session:
        session.configure_permutations(permutations)
        choices = []
        steps = []
        for iteration in range(options["iterations"]):
            chosen = cuda_search_permutation(options["random_seed"],
                options.get("iteration_offset", 0) + iteration, len(permutations))
            choices.append(chosen)
            session.select_permutation(chosen)
            steps.append(session.step())
        return session.result(), session.permutation_state, choices, steps


def _assert_equal(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types",
                 "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    assert all(array.dtype.kind != "O" for array in arrays.values())
    return arrays


@pytest.mark.parametrize("objective", ["RMSE", "CrossEntropy"])
@pytest.mark.parametrize("bootstrap", [{}, {"bootstrap_type": "MVS", "subsample": .6}])
def test_no_eval_forces_session_and_uses_absolute_cuda_search_schedule(
        metal_device, monkeypatch, objective, bootstrap):
    data = _problem(objective)
    bins, targets, features, borders, permutations, weights = data
    options = _options(objective=objective, **bootstrap)
    expected, state, choices, _ = _manual(data, options)
    assert set(choices) == {0, 1}
    assert not np.array_equal(state["predictions"][0], state["predictions"][-1])
    selected = []
    native_session = _native.Session

    class RecordingSession(native_session):
        def select_permutation(self, index):
            selected.append(index)
            return super().select_permutation(index)

    monkeypatch.setattr(_native, "Session", RecordingSession)
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("P>1 used the one-shot fast path"))
    actual = run_training(bins, targets, features, borders, permutation_bins=permutations,
                          sample_weight=weights, **options)
    assert selected == choices
    _assert_equal(actual, expected)
    np.testing.assert_array_equal(actual.predictions, state["predictions"][-1])
    np.testing.assert_array_equal(actual.evals_result["learn"][objective], expected.rmse[1:])
    assert actual.stats["permutation_count"] == 4
    assert actual.stats["estimation_permutation"] == 3
    assert actual.stats["permutation_schedule"] == "CatBoostMT19937_64"


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"},
    {"bootstrap_type": "Bayesian", "bagging_temperature": 1.5},
    {"bootstrap_type": "Bernoulli", "subsample": .55},
    {"bootstrap_type": "Poisson", "subsample": .6},
    {"bootstrap_type": "MVS", "subsample": .6},
    {"bootstrap_type": "MVS", "subsample": .6, "mvs_reg": .75},
])
def test_snapshot_restores_all_permutations_and_bootstrap_modes_exactly(metal_device, tmp_path, bootstrap):
    data = _problem()
    bins, targets, features, borders, permutations, weights = data
    options = _options(**bootstrap)
    expected, state, choices, _ = _manual(data, options)
    settings = dict(permutation_bins=permutations, sample_weight=weights,
                    eval_bins=permutations[-1], eval_targets=targets,
                    eval_weight=weights, use_best_model=False,
                    save_snapshot=True, snapshot_interval=0,
                    metadata={"test": "all permutation cursors"})
    path = tmp_path / "permutations.snapshot"
    partial = run_training(bins, targets, features, borders, snapshot_file=path,
                           **{**options, "iterations": 4}, **settings)
    saved = _read_arrays(path)
    assert saved["permutation_predictions"].shape == (4, len(targets))
    assert saved["permutation_predictions"].dtype == np.float32
    assert saved["permutation_mvs_lambdas"].shape == (4,)
    assert saved["permutation_mvs_lambdas"].dtype == np.float32
    assert saved["permutation_mvs_valid"].shape == (4,)
    assert saved["permutation_mvs_valid"].dtype == np.uint8
    np.testing.assert_array_equal(saved["permutation_predictions"][-1], partial.predictions)
    if bootstrap["bootstrap_type"] == "MVS" and bootstrap.get("mvs_reg") is None:
        assert np.ptp(saved["permutation_mvs_lambdas"]) > 1e-8
        np.testing.assert_array_equal(saved["permutation_mvs_valid"], np.ones(4, np.uint8))
    resumed = run_training(bins, targets, features, borders, snapshot_file=path, **options, **settings)
    uninterrupted = run_training(bins, targets, features, borders,
        **options, **{**settings, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, uninterrupted)
    assert resumed.evals_result == uninterrupted.evals_result
    assert resumed.stats["resumed_iterations"] == 4
    arrays = _read_arrays(path)
    for name in ("predictions", "mvs_lambdas", "mvs_valid"):
        np.testing.assert_array_equal(arrays["permutation_" + name], state[name], err_msg=name)
    header = json.loads(str(arrays["metadata"].item()))
    assert header["completed_iterations"] == options["iterations"]
    assert header["stats"]["bootstrap_state"]["iteration_offset"] == 17 + options["iterations"]
    np.testing.assert_array_equal(arrays["eval_predictions"], resumed.predictions)


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"}, {"bootstrap_type": "MVS", "subsample": .6},
])
def test_callback_interrupted_permutations_resume_the_original_total_tree_count(
        metal_device, tmp_path, bootstrap):
    data = _problem()
    bins, targets, features, borders, permutations, weights = data
    options = _options(**bootstrap)
    expected, final_state, _, _ = _manual(data, options)
    path = tmp_path / "callback.permutations.snapshot"
    settings = dict(permutation_bins=permutations, sample_weight=weights,
                    eval_bins=permutations[-1], eval_targets=targets,
                    eval_weight=weights, use_best_model=False,
                    save_snapshot=True, snapshot_file=path)
    seen = []

    def stop(info):
        seen.append(info.iteration)
        assert len(info.metrics["learn"]["RMSE"]) == info.iteration
        assert len(info.metrics["validation"]["RMSE"]) == info.iteration
        return info.iteration < 3

    stopped = run_training(bins, targets, features, borders, callback=stop, **options, **settings)
    assert seen == [1, 2, 3]
    assert stopped.stats["stop_reason"] == "callback"
    assert stopped.stats["iterations_trained"] == 3
    assert stopped.stopped_iteration == 2
    assert len(stopped.depths) == 3
    partial_arrays = _read_arrays(path)
    assert len(partial_arrays["depths"]) == 3
    assert partial_arrays["permutation_predictions"].shape == (4, len(targets))
    np.testing.assert_array_equal(partial_arrays["permutation_predictions"][-1], stopped.predictions)

    resumed_seen = []

    def observe(info):
        resumed_seen.append(info.iteration)
        return True

    resumed = run_training(bins, targets, features, borders, callback=observe, **options, **settings)
    assert resumed_seen == list(range(4, options["iterations"] + 1))
    assert resumed.stats["resumed_iterations"] == 3
    assert resumed.stats["iterations_trained"] == options["iterations"]
    assert resumed.stats["stop_reason"] == "iterations"
    _assert_equal(resumed, expected)
    np.testing.assert_array_equal(resumed.evals_result["learn"]["RMSE"], expected.rmse[1:])
    assert len(resumed.evals_result["validation"]["RMSE"]) == options["iterations"]
    arrays = _read_arrays(path)
    assert json.loads(str(arrays["metadata"].item()))["completed_iterations"] == options["iterations"]
    for name in ("predictions", "mvs_lambdas", "mvs_valid"):
        np.testing.assert_array_equal(arrays["permutation_" + name], final_state[name], err_msg=name)


@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"}, {"bootstrap_type": "MVS", "subsample": .6},
])
def test_completed_permutation_snapshot_preserves_all_state_without_opening_session(
        metal_device, monkeypatch, tmp_path, bootstrap):
    bins, targets, features, borders, permutations, weights = _problem(rows=67)
    path = tmp_path / "completed.permutations.snapshot"
    options = _options(iterations=4, **bootstrap)
    settings = dict(permutation_bins=permutations, sample_weight=weights,
                    eval_bins=permutations[-1], eval_targets=targets,
                    eval_weight=weights, use_best_model=False,
                    save_snapshot=True, snapshot_file=path)
    complete = run_training(bins, targets, features, borders, **options, **settings)
    before = _read_arrays(path)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Complete snapshot opened a session"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Complete snapshot started training"))
    from catboost_metal import _evaluation
    monkeypatch.setattr(_evaluation, "EvaluationCursor",
                        lambda *a, **k: pytest.fail("Complete snapshot opened an evaluation cursor"))
    resumed = run_training(bins, targets, features, borders, **options, **settings)
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == options["iterations"]
    assert resumed.stats["iterations_trained"] == options["iterations"]
    assert resumed.stats["kernel_dispatches"] == complete.stats["kernel_dispatches"]
    assert resumed.stats["validation"] == complete.stats["validation"]
    after = _read_arrays(path)
    assert set(after) == set(before)
    for name in before:
        if name != "metadata":
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
    header = json.loads(str(after["metadata"].item()))
    assert header["completed_iterations"] == options["iterations"]


def test_best_model_trimming_recomputes_on_final_estimation_bins(metal_device, tmp_path):
    p0 = np.array([[0, 0, 1, 1, 0, 0, 1, 1]], np.uint8)
    final = np.array([[0, 1, 0, 1, 0, 1, 0, 1]], np.uint8)
    permutations = (p0, 1 - p0, 1 - final, final)
    targets = np.where(final[0] > 0, 1., -1.).astype(np.float32)
    features = np.array([0], np.uint32)
    borders = np.array([0], np.uint32)
    path = tmp_path / "trimmed.snapshot"
    options = _options(iterations=8, depth=1, learning_rate=.5, l2_leaf_reg=0.,
                       bias=0., score_function="L2", leaf_estimation_iterations=1, random_strength=0.)
    result = run_training(p0, targets, features, borders, permutation_bins=permutations,
                          eval_bins=final, eval_targets=-targets,
                          early_stopping_rounds=2, use_best_model=True,
                          save_snapshot=True, snapshot_file=path, **options)
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert len(result.depths) == 1
    np.testing.assert_array_equal(result.predictions, targets * .5)
    assert not np.array_equal(result.predictions, np.where(p0[0] > 0, .5, -.5))
    assert result.evals_result["validation"]["RMSE"] == [1.5, 1.75, 1.875]
    saved = _read_arrays(path)
    assert saved["depths"].shape == (3,)
    np.testing.assert_array_equal(saved["permutation_predictions"][-1], targets * .875)
    np.testing.assert_array_equal(saved["predictions"], targets * .875)
    np.testing.assert_array_equal(saved["eval_predictions"], targets * .875)


def test_snapshot_fingerprints_nonzero_permutations_before_gpu(metal_device, monkeypatch, tmp_path):
    bins, targets, features, borders, permutations, weights = _problem(rows=67)
    path = tmp_path / "fingerprint.snapshot"
    run_training(bins, targets, features, borders, permutation_bins=permutations,
                 sample_weight=weights, save_snapshot=True, snapshot_file=path,
                 **_options(iterations=2))
    altered = list(permutations)
    altered[1] = altered[1].copy()
    altered[1][0, 0] = (int(altered[1][0, 0]) + 1) % 6
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Changed snapshot reached GPU"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Changed snapshot reached GPU"))
    with pytest.raises(ValueError, match="does not match"):
        run_training(bins, targets, features, borders, permutation_bins=tuple(altered),
                     sample_weight=weights, save_snapshot=True, snapshot_file=path, **_options())


@pytest.mark.parametrize("corruption", [
    "missing_predictions", "missing_mvs_lambdas", "missing_mvs_valid", "predictions_shape",
    "predictions_dtype", "predictions_nan", "predictions_pickle", "last_cursor_mismatch",
    "mvs_lambdas_shape", "mvs_lambdas_negative", "mvs_lambdas_nan", "mvs_valid_invalid",
    "mvs_valid_dtype", "mvs_valid_missing_adaptive_state",
])
def test_corrupt_all_permutation_snapshots_fail_before_gpu(metal_device, monkeypatch, tmp_path, corruption):
    bins, targets, features, borders, permutations, weights = _problem(rows=67)
    path = tmp_path / "corrupt.snapshot"
    options = _options(bootstrap_type="MVS", subsample=.6)
    common = dict(permutation_bins=permutations, sample_weight=weights,
                  save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read_arrays(path)
    if corruption.startswith("missing_"):
        arrays.pop("permutation_" + corruption.removeprefix("missing_"))
    elif corruption == "predictions_shape":
        arrays["permutation_predictions"] = arrays["permutation_predictions"][:-1]
    elif corruption == "predictions_dtype":
        arrays["permutation_predictions"] = arrays["permutation_predictions"].astype(np.float64)
    elif corruption == "predictions_nan":
        arrays["permutation_predictions"][1, 0] = np.nan
    elif corruption == "predictions_pickle":
        arrays["permutation_predictions"] = arrays["permutation_predictions"].astype(object)
    elif corruption == "last_cursor_mismatch":
        arrays["permutation_predictions"][-1, 0] += 1
    elif corruption == "mvs_lambdas_shape":
        arrays["permutation_mvs_lambdas"] = arrays["permutation_mvs_lambdas"][:-1]
    elif corruption == "mvs_lambdas_negative":
        arrays["permutation_mvs_lambdas"][1] = -1
    elif corruption == "mvs_lambdas_nan":
        arrays["permutation_mvs_lambdas"][1] = np.nan
    elif corruption == "mvs_valid_invalid":
        arrays["permutation_mvs_valid"][1] = 2
    elif corruption == "mvs_valid_dtype":
        arrays["permutation_mvs_valid"] = arrays["permutation_mvs_valid"].astype(np.float32)
    elif corruption == "mvs_valid_missing_adaptive_state":
        arrays["permutation_mvs_valid"][1] = 0
    with path.open("wb") as output:
        np.savez(output, **arrays)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Malformed snapshot reached GPU"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Malformed snapshot reached GPU"))
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, **options, **common)


def test_single_permutation_keeps_the_native_train_fast_path(metal_device, monkeypatch):
    bins, targets, features, borders, _, weights = _problem(rows=67)
    options = _options(iterations=3)
    expected = _native.train(bins, targets, features, borders, sample_weight=weights, **options)
    calls = []

    def direct(*args, **kwargs):
        calls.append(kwargs)
        assert "permutation_bins" not in kwargs
        return copy.deepcopy(expected)

    monkeypatch.setattr(_native, "train", direct)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("P=1 bypassed the fast path"))
    result = run_training(bins, targets, features, borders, permutation_bins=(bins,),
                          sample_weight=weights, **options)
    assert len(calls) == 1
    _assert_equal(result, expected)
