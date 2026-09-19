"""Metal controller feature penalties and portable used-feature snapshots.

Independent native sessions supply the model/state comparison. These tests do
not train CatBoost CPU models or replace the native scoring formula with mocks.
"""

import copy
import platform

import numpy as np
import pytest

from catboost_metal import _native
from catboost_metal._data import cuda_search_permutation
from catboost_metal._training import run_training


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Feature penalty tests must not train CatBoost on CPU")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Feature penalty lifecycle tests require Apple Silicon")
    assert _native.device_info()["backend"] == "Metal"


def _problem(permutation_count=1, rows=259):
    rng = np.random.default_rng(12974)
    bins = rng.integers(0, 6, size=(3, rows), dtype=np.uint8)
    matrices = [bins.copy()]
    for _ in range(permutation_count - 1):
        current = bins.copy()
        for column in current:
            rng.shuffle(column)
        matrices.append(current)
    targets = (1.3 * (bins[0] > 2) - 1.1 * (bins[1] > 3)
               + .4 * (bins[2] > 1) + rng.normal(0, .1, rows)).astype(np.float32)
    weights = rng.uniform(.2, 2, rows).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    return bins, targets, features, borders, tuple(matrices), weights


def _options(**overrides):
    options = dict(iterations=9, depth=2, learning_rate=.2, l2_leaf_reg=2,
                   bias=.03125, score_function="Cosine", objective="RMSE",
                   leaf_estimation_iterations=3, random_seed=671,
                   iteration_offset=17, random_strength=.3)
    options.update(overrides)
    return options


def _penalties(kind="combined"):
    return dict(ctr_unique_values=(None if kind == "weights_only" else
                                  np.array([0, 13, 57], np.uint32)),
                model_size_reg=.7,
                feature_weights=(None if kind == "ctr_only" else
                                 np.array([.2, 2., .5], np.float32)))


def _manual(data, options, penalties):
    bins, targets, features, borders, matrices, weights = data
    with _native.Session(bins, targets, features, borders, sample_weight=weights, **options) as session:
        if len(matrices) > 1:
            session.configure_permutations(matrices)
        counts = penalties["ctr_unique_values"]
        session.configure_feature_penalties(
            np.zeros(bins.shape[0], np.uint32) if counts is None else counts,
            model_size_reg=penalties["model_size_reg"], feature_weights=penalties["feature_weights"])
        states = []
        for index in range(options["iterations"]):
            if len(matrices) > 1:
                session.select_permutation(cuda_search_permutation(options["random_seed"],
                    options.get("iteration_offset", 0) + index, len(matrices)))
            session.step()
            states.append(session.feature_penalty_state["used_features"])
        return session.result(), states, session.permutation_state


def _assert_equal(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types",
                 "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    assert all(array.dtype.kind != "O" for array in arrays.values())
    return arrays


@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("kind", ["ctr_only", "weights_only", "combined"])
def test_controller_penalties_match_manual_native_session(metal_device, monkeypatch, permutation_count, kind):
    data = _problem(permutation_count)
    bins, targets, features, borders, matrices, weights = data
    options, penalties = _options(), _penalties(kind)
    expected, states, _ = _manual(data, options, penalties)
    configured = []
    original = _native.Session

    class RecordingSession(original):
        def configure_feature_penalties(self, counts, model_size_reg=.5, feature_weights=None, used_features=None):
            assert self.completed_iterations == 0
            configured.append((np.asarray(counts).copy(), model_size_reg,
                               None if feature_weights is None else np.asarray(feature_weights).copy(),
                               used_features))
            return super().configure_feature_penalties(counts, model_size_reg, feature_weights, used_features)

    monkeypatch.setattr(_native, "Session", RecordingSession)
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Configured penalties used one-shot train"))
    actual = run_training(bins, targets, features, borders, permutation_bins=matrices,
                          sample_weight=weights, **options, **penalties)
    _assert_equal(actual, expected)
    assert len(configured) == 1
    counts = np.zeros(3, np.uint32) if penalties["ctr_unique_values"] is None else penalties["ctr_unique_values"]
    np.testing.assert_array_equal(configured[0][0], counts)
    assert configured[0][1] == pytest.approx(penalties["model_size_reg"])
    if penalties["feature_weights"] is not None:
        np.testing.assert_array_equal(configured[0][2], penalties["feature_weights"])
    assert configured[0][3] is None
    if kind == "weights_only":
        np.testing.assert_array_equal(states[-1], np.zeros(3, np.uint8))
    else:
        assert states[-1].any()
        assert states[-1][0] == 0
        # Only the shared tree structure changes these flags; extra estimation
        # permutations do not each have separate used-feature histories.
        selected = set()
        for depth, tree in zip(actual.depths, actual.split_features):
            selected.update(int(feature) for feature in tree[:depth] if counts[feature])
        assert all(states[-1][feature] == 1 for feature in selected)


@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("bootstrap", [{}, {"bootstrap_type": "MVS", "subsample": .6}])
def test_penalty_and_permutation_snapshots_resume_exactly(
        metal_device, monkeypatch, tmp_path, permutation_count, bootstrap):
    data = _problem(permutation_count)
    bins, targets, features, borders, matrices, weights = data
    options, penalties = _options(**bootstrap), _penalties()
    expected, states, permutation_state = _manual(data, options, penalties)
    path = tmp_path / "penalties.snapshot"
    common = dict(permutation_bins=matrices, sample_weight=weights,
                  eval_bins=matrices[-1], eval_targets=targets, eval_weight=weights,
                  use_best_model=False, save_snapshot=True, snapshot_file=path,
                  snapshot_interval=0, **penalties)
    run_training(bins, targets, features, borders, **{**options, "iterations": 4}, **common)
    partial = _read_arrays(path)
    used = partial["feature_penalty_used_features"]
    assert used.shape == (bins.shape[0],) and used.dtype == np.uint8
    np.testing.assert_array_equal(used, states[3])
    assert used.any()
    restored = []
    original = _native.Session

    class RecordingSession(original):
        def configure_feature_penalties(self, counts, model_size_reg=.5, feature_weights=None, used_features=None):
            assert self.completed_iterations == 0
            restored.append(None if used_features is None else np.asarray(used_features).copy())
            return super().configure_feature_penalties(counts, model_size_reg, feature_weights, used_features)

    with monkeypatch.context() as patch:
        patch.setattr(_native, "Session", RecordingSession)
        resumed = run_training(bins, targets, features, borders, **options, **common)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0], used)
    complete = run_training(bins, targets, features, borders, **options,
                            **{**common, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == 4
    arrays = _read_arrays(path)
    np.testing.assert_array_equal(arrays["feature_penalty_used_features"], states[-1])
    if permutation_count > 1:
        for name in ("predictions", "mvs_lambdas", "mvs_valid"):
            np.testing.assert_array_equal(arrays["permutation_" + name], permutation_state[name], err_msg=name)


@pytest.mark.parametrize("change", ["counts", "weights", "strength"])
def test_penalty_config_changes_invalidate_snapshot_before_gpu(metal_device, monkeypatch, tmp_path, change):
    bins, targets, features, borders, matrices, weights = _problem(4, rows=67)
    penalties = _penalties()
    path = tmp_path / "fingerprint.snapshot"
    common = dict(permutation_bins=matrices, sample_weight=weights, save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **_options(iterations=2), **common, **penalties)
    changed = copy.deepcopy(penalties)
    if change == "counts":
        changed["ctr_unique_values"][1] += 1
    elif change == "weights":
        changed["feature_weights"][1] += .25
    else:
        changed["model_size_reg"] += .25
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Changed penalties reached GPU"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Changed penalties reached GPU"))
    with pytest.raises(ValueError, match="does not match"):
        run_training(bins, targets, features, borders, **_options(), **common, **changed)


@pytest.mark.parametrize("corruption", ["missing", "short", "matrix", "float", "signed", "nonbinary", "pickle",
                                         "used_ctr_cleared", "numeric_marked_used"])
def test_invalid_penalty_snapshot_flags_are_rejected_before_gpu(metal_device, monkeypatch, tmp_path, corruption):
    bins, targets, features, borders, matrices, weights = _problem(4, rows=67)
    path = tmp_path / "invalid.snapshot"
    common = dict(permutation_bins=matrices, sample_weight=weights,
                  save_snapshot=True, snapshot_file=path, **_penalties())
    run_training(bins, targets, features, borders, **_options(iterations=2), **common)
    arrays = _read_arrays(path)
    name = "feature_penalty_used_features"
    if corruption == "missing":
        arrays.pop(name)
    elif corruption == "short":
        arrays[name] = arrays[name][:-1]
    elif corruption == "matrix":
        arrays[name] = arrays[name].reshape(1, -1)
    elif corruption == "float":
        arrays[name] = arrays[name].astype(np.float32)
    elif corruption == "signed":
        arrays[name] = arrays[name].astype(np.int8)
    elif corruption == "nonbinary":
        arrays[name][1] = 2
    elif corruption == "pickle":
        arrays[name] = arrays[name].astype(object)
    elif corruption == "used_ctr_cleared":
        selected_ctrs = [int(feature) for depth, tree in zip(arrays["depths"], arrays["split_features"])
                         for feature in tree[:depth] if common["ctr_unique_values"][feature]]
        assert selected_ctrs
        arrays[name][selected_ctrs[0]] = 0
    elif corruption == "numeric_marked_used":
        arrays[name][0] = 1
    with path.open("wb") as output:
        np.savez(output, **arrays)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Malformed penalty flags reached GPU"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Malformed penalty flags reached GPU"))
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, **_options(), **common)


def test_completed_penalty_snapshot_keeps_used_flags_without_gpu(metal_device, monkeypatch, tmp_path):
    bins, targets, features, borders, matrices, weights = _problem(4, rows=67)
    path = tmp_path / "completed.snapshot"
    common = dict(permutation_bins=matrices, sample_weight=weights,
                  save_snapshot=True, snapshot_file=path, **_penalties())
    options = _options(iterations=3)
    expected = run_training(bins, targets, features, borders, **options, **common)
    before = _read_arrays(path)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Complete penalties opened a session"))
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Complete penalties restarted training"))
    actual = run_training(bins, targets, features, borders, **options, **common)
    _assert_equal(actual, expected)
    after = _read_arrays(path)
    for name in ("feature_penalty_used_features", "permutation_predictions",
                 "permutation_mvs_lambdas", "permutation_mvs_valid"):
        np.testing.assert_array_equal(after[name], before[name], err_msg=name)


def test_old_numeric_fast_path_does_not_configure_penalties(metal_device, monkeypatch):
    bins, targets, features, borders, _, weights = _problem(rows=67)
    options = _options(iterations=3)
    expected = _native.train(bins, targets, features, borders, sample_weight=weights, **options)
    calls = []

    def train(*args, **kwargs):
        calls.append(kwargs)
        assert not set(kwargs) & {"ctr_unique_values", "model_size_reg", "feature_weights"}
        return copy.deepcopy(expected)

    monkeypatch.setattr(_native, "train", train)
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Unconfigured numeric path opened a session"))
    actual = run_training(bins, targets, features, borders, sample_weight=weights, **options)
    assert len(calls) == 1
    _assert_equal(actual, expected)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("bootstrap", [
    {"bootstrap_type": "No"}, {"bootstrap_type": "Bayesian", "bagging_temperature": 1.5},
    {"bootstrap_type": "Bernoulli", "subsample": .6},
    {"bootstrap_type": "Poisson", "subsample": .7},
])
def test_multiclass_penalty_and_sampler_snapshots_restore_exactly(
        metal_device, monkeypatch, tmp_path, objective, permutation_count, bootstrap):
    from catboost_metal import _multiclass

    bins, _, features, borders, matrices, weights = _problem(permutation_count, rows=67)
    targets = ((bins[0].astype(np.uint32) + 2 * bins[1]) % 3).astype(np.float32)
    options = _options(iterations=7, objective=objective, classes=3, score_function="L2",
                       bias=np.array([.1, -.2, .37], np.float32), **bootstrap)
    penalties = _penalties()
    with _multiclass.Session(bins, targets, features, borders, sample_weight=weights, **options) as native:
        if permutation_count > 1:
            native.configure_permutations(matrices)
        native.configure_feature_penalties(penalties["ctr_unique_values"],
            model_size_reg=penalties["model_size_reg"], feature_weights=penalties["feature_weights"])
        for iteration in range(options["iterations"]):
            if permutation_count > 1:
                native.select_permutation(cuda_search_permutation(options["random_seed"],
                    options["iteration_offset"] + iteration, permutation_count))
            native.step()
        expected = native.result()
        used_final = native.feature_penalty_state["used_features"]
        permutation_final = native.permutation_state if permutation_count > 1 else None
        optimizer_final = native.optimization_predictions()
    path = tmp_path / "multiclass.penalties.snapshot"
    common = dict(permutation_bins=matrices, sample_weight=weights,
                  eval_bins=matrices[-1], eval_targets=targets, eval_weight=weights,
                  use_best_model=False, save_snapshot=True, snapshot_file=path, **penalties)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    saved_used = _read_arrays(path)["feature_penalty_used_features"]
    assert saved_used.dtype == np.uint8 and saved_used.shape == (bins.shape[0],)
    restored = []
    original = _multiclass.Session

    class RecordingSession(original):
        def configure_feature_penalties(self, counts, model_size_reg=.5, feature_weights=None, used_features=None):
            assert self.completed_iterations == 0
            restored.append(None if used_features is None else np.asarray(used_features).copy())
            return super().configure_feature_penalties(counts, model_size_reg, feature_weights, used_features)

    with monkeypatch.context() as patch:
        patch.setattr(_multiclass, "Session", RecordingSession)
        resumed = run_training(bins, targets, features, borders, **options, **common)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0], saved_used)
    complete = run_training(bins, targets, features, borders, **options,
                            **{**common, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == 2
    arrays = _read_arrays(path)
    np.testing.assert_array_equal(arrays["feature_penalty_used_features"], used_final)
    if permutation_count > 1:
        for name, value in permutation_final.items():
            np.testing.assert_array_equal(arrays["permutation_" + name], value, err_msg=name)
    else:
        np.testing.assert_array_equal(arrays["optimization_predictions"], optimizer_final)
