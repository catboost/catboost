"""Ordered controller routing and exact prefix-cursor snapshot continuation."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import _native, _ordered
from catboost_metal._training import run_training, _fingerprint


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Ordered lifecycle tests must not train CatBoost on CPU")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Ordered lifecycle tests require Apple Silicon Metal")


def _problem(objective="RMSE", rows=127):
    rng = np.random.default_rng(177394)
    bins = rng.integers(0, 8, size=(3, rows), dtype=np.uint8)
    signal = (.41 * bins[0].astype(np.float32) - .29 * bins[1] + .37 * (bins[2] > 3)
              + rng.normal(0, .1, rows) - .6)
    if objective == "RMSE":
        targets = signal
    else:
        probabilities = 1 / (1 + np.exp(-signal))
        targets = probabilities if objective == "CrossEntropy" else (rng.random(rows) < probabilities)
    targets = np.asarray(targets, np.float32)
    weights = rng.uniform(.2, 2., rows).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 3)
    return bins, targets, features, borders, weights


def _options(objective="RMSE", **overrides):
    result = dict(iterations=7, depth=2, learning_rate=.17, l2_leaf_reg=2.5,
                  bias=.03125, objective=objective, score_function="Cosine",
                  leaf_estimation_method="Newton", leaf_estimation_iterations=3,
                  permutation_count=1, min_fold_size=16, fold_len_multiplier=1.7,
                  fold_size_loss_normalization=True, random_seed=671, iteration_offset=17)
    result.update(overrides)
    return result


def _manual(data, options):
    bins, targets, features, borders, weights = data
    second_state = None
    with _ordered.Session(bins, targets, features, borders, sample_weight=weights, **options) as session:
        for iteration in range(options["iterations"]):
            session.step()
            if iteration == 1:
                second_state = session.state()
        return session.result(), session.state(), second_state


def _assert_equal(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types",
                 "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def _read(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    assert all(array.dtype.kind != "O" for array in result.values())
    return result


def _assert_state(arrays, state):
    np.testing.assert_array_equal(arrays["ordered_descriptors"], state["descriptors"])
    np.testing.assert_array_equal(arrays["ordered_cursors"], state["cursors"])
    header = json.loads(str(arrays["metadata"].item()))["ordered_state"]
    for name in ("version", "fingerprint", "iteration_offset", "mvs_lambda"):
        assert header[name] == state[name]
    if "selection_rng" in state:
        np.testing.assert_array_equal(arrays["ordered_selection_rng_words"], state["selection_rng"]["words"])
        assert header["selection_rng"] == {key: value for key, value in state["selection_rng"].items() if key != "words"}


def _forbid_gpu(monkeypatch):
    from catboost_metal import _evaluation

    for module in (_native, _ordered):
        monkeypatch.setattr(module, "Session", lambda *a, **k: pytest.fail("Invalid snapshot opened a GPU session"))
        monkeypatch.setattr(module, "train", lambda *a, **k: pytest.fail("Invalid snapshot started GPU training"))
    monkeypatch.setattr(_evaluation, "EvaluationCursor",
                        lambda *a, **k: pytest.fail("Invalid snapshot opened GPU validation"))


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("score,method", [("Cosine", "Newton"), ("NewtonCosine", "Gradient")])
def test_ordered_controller_matches_native_with_weights_and_folds(
        metal_device, monkeypatch, objective, permutations, score, method):
    data = _problem(objective)
    bins, targets, features, borders, weights = data
    options = _options(objective, permutation_count=permutations,
                       score_function=score, leaf_estimation_method=method)
    expected, state, _ = _manual(data, options)
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Ordered used Plain GPU training"))
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Ordered used a Plain GPU session"))
    actual = run_training(bins, targets, features, borders, boosting_type="Ordered",
                          sample_weight=weights, **options)
    _assert_equal(actual, expected)
    assert actual.stats["boosting_type"] == "Ordered"
    assert actual.stats["device"].startswith("Apple")
    assert actual.stats["kernel_dispatches"] > 0
    assert actual.stats["permutation_count"] == permutations
    assert actual.stats["ordered_tasks"] > 1
    assert len(state["cursors"]) > len(targets)
    np.testing.assert_array_equal(actual.evals_result["learn"][objective], expected.rmse[1:])


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("score,method", [("Cosine", "Newton"), ("NewtonCosine", "Gradient")])
def test_ordered_snapshot_restores_every_fold_cursor_from_two_to_seven_trees(
        metal_device, monkeypatch, tmp_path, objective, permutations, score, method):
    data = _problem(objective)
    bins, targets, features, borders, weights = data
    options = _options(objective, permutation_count=permutations,
                       score_function=score, leaf_estimation_method=method)
    expected, final_state, second_state = _manual(data, options)
    path = tmp_path / "ordered.snapshot"
    common = dict(boosting_type="Ordered", sample_weight=weights, eval_bins=bins,
                  eval_targets=targets, eval_weight=weights, use_best_model=False,
                  save_snapshot=True, snapshot_file=path, snapshot_interval=0)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    _assert_state(_read(path), second_state)
    restored = []
    original = _ordered.Session

    class RecordingSession(original):
        def __init__(self, *args, **kwargs):
            restored.append(kwargs.get("initial_state"))
            super().__init__(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(_ordered, "Session", RecordingSession)
        resumed = run_training(bins, targets, features, borders, **options, **common)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0]["cursors"], second_state["cursors"])
    np.testing.assert_array_equal(restored[0]["descriptors"], second_state["descriptors"])
    assert restored[0]["iteration_offset"] == 19
    uninterrupted = run_training(bins, targets, features, borders, **options,
                                 **{**common, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, uninterrupted)
    assert resumed.evals_result == uninterrupted.evals_result
    assert resumed.stats["resumed_iterations"] == 2
    _assert_state(_read(path), final_state)


@pytest.mark.parametrize("permutations", [1, 4])
def test_ordered_early_stopping_saves_untrimmed_fold_state(metal_device, tmp_path, permutations):
    bins = np.zeros((1, 67), np.uint8)
    targets = np.ones(67, np.float32)
    empty = np.empty(0, np.uint32)
    path = tmp_path / "best.snapshot"
    options = _options(depth=0, learning_rate=.5, l2_leaf_reg=0., bias=0.,
                       leaf_estimation_iterations=1, permutation_count=permutations)
    common = dict(boosting_type="Ordered", eval_bins=bins, eval_targets=-targets,
                  early_stopping_rounds=2, use_best_model=True,
                  save_snapshot=True, snapshot_file=path)
    result = run_training(bins, targets, empty, empty, **options, **common)
    assert len(result.depths) == 1
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert result.stats["stop_reason"] == "early_stopping"
    np.testing.assert_array_equal(result.predictions, targets * .5)
    assert result.evals_result["validation"]["RMSE"] == [1.5, 1.75, 1.875]
    arrays = _read(path)
    assert len(arrays["depths"]) == 3
    np.testing.assert_array_equal(arrays["predictions"], targets * .875)
    assert len(arrays["ordered_cursors"]) > len(targets)
    resumed = run_training(bins, targets, empty, empty, **options, **common)
    _assert_equal(resumed, result)
    np.testing.assert_array_equal(_read(path)["ordered_cursors"], arrays["ordered_cursors"])


@pytest.mark.parametrize("permutations", [1, 4])
def test_callback_stopped_ordered_snapshot_resumes_total_tree_count(metal_device, tmp_path, permutations):
    data = _problem()
    bins, targets, features, borders, weights = data
    options = _options(permutation_count=permutations)
    expected, state, _ = _manual(data, options)
    path = tmp_path / "callback.snapshot"
    common = dict(boosting_type="Ordered", sample_weight=weights,
                  save_snapshot=True, snapshot_file=path)
    observed = []

    def stop(info):
        observed.append(info.iteration)
        return info.iteration < 2

    first = run_training(bins, targets, features, borders, callback=stop, **options, **common)
    assert observed == [1, 2]
    assert first.stats["stop_reason"] == "callback"
    assert len(first.depths) == 2
    resumed = run_training(bins, targets, features, borders, **options, **common)
    _assert_equal(resumed, expected)
    assert resumed.stats["resumed_iterations"] == 2
    assert len(resumed.depths) == 7
    _assert_state(_read(path), state)


@pytest.mark.parametrize("corruption", [
    "missing_state", "missing_cursors", "missing_descriptors", "short_cursors", "float64_cursors",
    "nan_cursors", "pickle_cursors", "descriptor_shape", "descriptor_dtype", "descriptor_offset",
    "version", "fingerprint", "iteration_offset",
])
def test_invalid_ordered_snapshot_state_fails_before_gpu(metal_device, monkeypatch, tmp_path, corruption):
    bins, targets, features, borders, weights = _problem(rows=67)
    path = tmp_path / "invalid.snapshot"
    options = _options(permutation_count=4)
    common = dict(boosting_type="Ordered", sample_weight=weights, save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read(path)
    header = json.loads(str(arrays["metadata"].item()))
    if corruption == "missing_state":
        header.pop("ordered_state")
    elif corruption == "missing_cursors":
        arrays.pop("ordered_cursors")
    elif corruption == "missing_descriptors":
        arrays.pop("ordered_descriptors")
    elif corruption == "short_cursors":
        arrays["ordered_cursors"] = arrays["ordered_cursors"][:-1]
    elif corruption == "float64_cursors":
        arrays["ordered_cursors"] = arrays["ordered_cursors"].astype(np.float64)
    elif corruption == "nan_cursors":
        arrays["ordered_cursors"][0] = np.nan
    elif corruption == "pickle_cursors":
        arrays["ordered_cursors"] = arrays["ordered_cursors"].astype(object)
    elif corruption == "descriptor_shape":
        arrays["ordered_descriptors"] = arrays["ordered_descriptors"][:, :3]
    elif corruption == "descriptor_dtype":
        arrays["ordered_descriptors"] = arrays["ordered_descriptors"].astype(np.float32)
    elif corruption == "descriptor_offset":
        arrays["ordered_descriptors"][0, 2] += 1
    elif corruption == "version":
        header["ordered_state"]["version"] += 1
    elif corruption == "fingerprint":
        header["ordered_state"]["fingerprint"] = "invalid"
    elif corruption == "iteration_offset":
        header["ordered_state"]["iteration_offset"] += 1
    arrays["metadata"] = np.asarray(json.dumps(header))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, **options, **common)


@pytest.mark.parametrize("corruption", ["missing_words", "short_words", "wrong_dtype", "zero_words",
                                         "index", "version", "bootstrap", "completed", "missing_metadata"])
def test_ordered_selection_rng_snapshot_is_validated_before_gpu(
        metal_device, monkeypatch, tmp_path, corruption):
    bins, targets, features, borders, weights = _problem(rows=67)
    options = _options(iterations=3, permutation_count=4)
    path = tmp_path / "invalid.rng.snapshot"
    common = dict(boosting_type="Ordered", sample_weight=weights,
                  save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read(path)
    header = json.loads(str(arrays["metadata"].item()))
    state = header["ordered_state"]
    rng = state["selection_rng"]
    if corruption == "missing_words":
        arrays.pop("ordered_selection_rng_words")
    elif corruption == "short_words":
        arrays["ordered_selection_rng_words"] = arrays["ordered_selection_rng_words"][:-1]
    elif corruption == "wrong_dtype":
        arrays["ordered_selection_rng_words"] = arrays["ordered_selection_rng_words"].astype(np.float64)
    elif corruption == "zero_words":
        arrays["ordered_selection_rng_words"][:] = 0
    elif corruption == "missing_metadata":
        state.pop("selection_rng")
    else:
        key, value = {"index": ("index", 313), "version": ("version", 2),
                      "bootstrap": ("bootstrap_initialized", False),
                      "completed": ("completed_iterations", rng["completed_iterations"] + 1)}[corruption]
        rng[key] = value
    header["ordered_state_checksum"] = _fingerprint(
        {name.removeprefix("ordered_"): value for name, value in arrays.items()
         if name in ("ordered_descriptors", "ordered_cursors", "ordered_selection_rng_words")}, state)
    arrays["metadata"] = np.asarray(json.dumps(header))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="(?i)rng|selection"):
        run_training(bins, targets, features, borders, **options, **common)


def test_ordered_large_pool_snapshot_uses_cuda_block_permutations(metal_device, tmp_path):
    rows = 50_003
    bins = np.random.default_rng(981).integers(0, 2, size=(1, rows), dtype=np.uint8)
    targets = (2 * bins[0].astype(np.float32) - 1)
    features = np.asarray([0], np.uint32)
    borders = np.asarray([0], np.uint32)
    options = _options(iterations=3, permutation_count=4, depth=1, fold_permutation_block=64)
    common = dict(boosting_type="Ordered", save_snapshot=True,
                  snapshot_file=tmp_path / "block.snapshot")
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    resumed = run_training(bins, targets, features, borders, **options, **common)
    full = run_training(bins, targets, features, borders, **options, boosting_type="Ordered")
    _assert_equal(resumed, full)
    assert len(np.unique(resumed.predictions)) == 2


def test_completed_ordered_snapshot_keeps_fold_state_without_gpu(metal_device, monkeypatch, tmp_path):
    bins, targets, features, borders, weights = _problem(rows=67)
    path = tmp_path / "completed.snapshot"
    options = _options(iterations=3, permutation_count=4)
    common = dict(boosting_type="Ordered", sample_weight=weights, save_snapshot=True, snapshot_file=path)
    expected = run_training(bins, targets, features, borders, **options, **common)
    before = _read(path)
    _forbid_gpu(monkeypatch)
    actual = run_training(bins, targets, features, borders, **options, **common)
    _assert_equal(actual, expected)
    after = _read(path)
    for name in before:
        if name != "metadata":
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def _exercise_extended_snapshot(data, options, path):
    bins, targets, features, borders, weights = data
    expected, final_state, second_state = _manual(data, options)
    common = dict(boosting_type="Ordered", sample_weight=weights, eval_bins=bins,
                  eval_targets=targets, eval_weight=weights, use_best_model=False,
                  save_snapshot=True, snapshot_file=path, snapshot_interval=0)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    _assert_state(_read(path), second_state)
    resumed = run_training(bins, targets, features, borders, **options, **common)
    uninterrupted = run_training(bins, targets, features, borders, **options,
                                 **{**common, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, uninterrupted)
    assert resumed.evals_result == uninterrupted.evals_result
    assert resumed.stats["resumed_iterations"] == 2
    assert resumed.stats["search_permutations"] == uninterrupted.stats["search_permutations"]
    assert resumed.stats["bootstrap_state"] == uninterrupted.stats["bootstrap_state"]
    arrays = _read(path)
    _assert_state(arrays, final_state)
    return json.loads(str(arrays["metadata"].item()))["ordered_state"]


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("observations", ["TestOnly", "LearnAndTest"])
@pytest.mark.parametrize("sampling", [
    pytest.param({"bootstrap_type": "No"}, id="No"),
    pytest.param({"bootstrap_type": "Bayesian", "bagging_temperature": 1.5}, id="Bayesian"),
    pytest.param({"bootstrap_type": "Bernoulli", "subsample": .55}, id="Bernoulli"),
    pytest.param({"bootstrap_type": "Poisson", "subsample": .65}, id="Poisson"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .6}, id="AutoMVS"),
    pytest.param({"bootstrap_type": "MVS", "subsample": .6, "mvs_reg": .75}, id="FixedMVS"),
])
def test_ordered_all_samplers_and_score_noise_resume_exactly(
        metal_device, tmp_path, permutations, observations, sampling):
    options = _options(permutation_count=permutations, random_strength=.8,
                       observations_to_bootstrap=observations, **sampling)
    state = _exercise_extended_snapshot(_problem(), options, tmp_path / "sampling.snapshot")
    if sampling["bootstrap_type"] == "MVS" and sampling.get("mvs_reg") is None:
        assert isinstance(state["mvs_lambda"], float) and state["mvs_lambda"] >= 0
    else:
        assert state["mvs_lambda"] is None


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("objective,parameter,method", [
    ("Poisson", None, "Newton"), ("Huber", .8, "Newton"),
    ("Expectile", .7, "Newton"), ("Lq", 2.7, "Newton"),
    ("Tweedie", 1.3, "Newton"), ("LogLinQuantile", .65, "Gradient"),
    ("Quantile", .65, "Gradient"), ("MAE", None, "Gradient"),
    ("MAPE", None, "Gradient"),
])
def test_ordered_additional_objective_snapshots_are_exact(
        metal_device, tmp_path, permutations, objective, parameter, method):
    bins, targets, features, borders, weights = _problem()
    if objective in ("Poisson", "Tweedie", "LogLinQuantile"):
        targets = np.exp(.5 * targets).astype(np.float32)
    options = _options(objective, permutation_count=permutations, learning_rate=.07,
                       objective_param=parameter, leaf_estimation_method=method,
                       score_function="NewtonCosine", leaf_estimation_iterations=2)
    state = _exercise_extended_snapshot((bins, targets, features, borders, weights),
                                        options, tmp_path / "objective.snapshot")
    assert state["mvs_lambda"] is None


@pytest.mark.parametrize("corruption", ["missing", "null", "negative", "boolean", "string", "nan", "mismatch"])
def test_ordered_invalid_adaptive_mvs_state_fails_before_gpu(metal_device, monkeypatch, tmp_path, corruption):
    bins, targets, features, borders, weights = _problem(rows=67)
    options = _options(permutation_count=4, bootstrap_type="MVS", subsample=.6, random_strength=.8)
    path = tmp_path / "invalid.mvs.snapshot"
    common = dict(boosting_type="Ordered", sample_weight=weights,
                  save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read(path)
    header = json.loads(str(arrays["metadata"].item()))
    state = header["ordered_state"]
    assert state["mvs_lambda"] is not None
    if corruption == "missing":
        state.pop("mvs_lambda")
    else:
        state["mvs_lambda"] = {
            "null": None, "negative": -1, "boolean": True, "string": "invalid",
            "nan": float("nan"), "mismatch": state["mvs_lambda"] + 1,
        }[corruption]
    # Preserve the checksum for valid JSON mutations to exercise semantic MVS
    # checks independently of archive integrity. NaN is not canonical JSON.
    if corruption != "nan":
        header["ordered_state_checksum"] = _fingerprint(
            {name.removeprefix("ordered_"): value for name, value in arrays.items()
             if name in ("ordered_descriptors", "ordered_cursors", "ordered_selection_rng_words")}, state)
    arrays["metadata"] = np.asarray(json.dumps(header))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, **options, **common)
