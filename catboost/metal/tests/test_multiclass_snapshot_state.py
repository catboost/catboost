"""Exact class-major multiclass optimizer snapshots, including nonzero gauges."""

import platform

import numpy as np
import pytest

from catboost_metal import _multiclass, _native
from catboost_metal._training import run_training


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Multiclass optimizer-state tests must not train CatBoost on CPU")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Multiclass optimizer-state tests require Apple Silicon")
    assert _native.device_info()["backend"] == "Metal"


def _problem(permutations=1, rows=131):
    rng = np.random.default_rng(197326)
    bins = rng.integers(0, 6, size=(3, rows), dtype=np.uint8)
    matrices = [bins.copy()]
    for _ in range(permutations - 1):
        current = bins.copy()
        for column in current:
            rng.shuffle(column)
        matrices.append(current)
    targets = ((2 * bins[0].astype(np.uint32) + bins[1]) % 3).astype(np.float32)
    weights = rng.uniform(.125, 2.75, rows).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 5)
    borders = np.tile(np.arange(5, dtype=np.uint32), 3)
    return bins, targets, features, borders, tuple(matrices), weights


def _options(objective="MultiClass", **overrides):
    result = dict(iterations=8, depth=2, learning_rate=.23, l2_leaf_reg=2.75,
                  bias=np.array([1.1234567, -2.7654321, .9182736], np.float32),
                  classes=3, score_function="L2", objective=objective,
                  leaf_estimation_iterations=3, random_seed=9137,
                  iteration_offset=11, random_strength=.35)
    result.update(overrides)
    return result


def _read(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {name: archive[name].copy() for name in archive.files}
    assert all(array.dtype.kind != "O" for array in result.values())
    return result


def _write(path, arrays):
    with path.open("wb") as output:
        np.savez(output, **arrays)


def _assert_equal(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types",
                 "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def _forbid_gpu(monkeypatch):
    from catboost_metal import _evaluation

    for module in (_native, _multiclass):
        monkeypatch.setattr(module, "Session", lambda *a, **k: pytest.fail("Snapshot opened a GPU session"))
        monkeypatch.setattr(module, "train", lambda *a, **k: pytest.fail("Snapshot started GPU training"))
    monkeypatch.setattr(_evaluation, "EvaluationCursor",
                        lambda *a, **k: pytest.fail("Snapshot opened a GPU validation cursor"))


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("bootstrap", [{"bootstrap_type": "No"},
                                        {"bootstrap_type": "Bernoulli", "subsample": .7}])
def test_p1_nonzero_vector_bias_restores_exact_optimizer_bits(metal_device, tmp_path, objective, bootstrap):
    bins, targets, features, borders, _, weights = _problem()
    options = _options(objective, **bootstrap)
    partial_native_state = None
    with _multiclass.Session(bins, targets, features, borders, sample_weight=weights, **options) as native:
        for iteration in range(options["iterations"]):
            native.step()
            if iteration == 2:
                partial_native_state = native.optimization_predictions()
        expected = native.result()
        final_native_state = native.optimization_predictions()
    path = tmp_path / "resume.snapshot"
    complete_path = tmp_path / "complete.snapshot"
    settings = dict(sample_weight=weights, eval_bins=bins, eval_targets=targets,
                    eval_weight=weights, use_best_model=False, save_snapshot=True,
                    snapshot_interval=0)
    partial = run_training(bins, targets, features, borders, snapshot_file=path,
                           **{**options, "iterations": 3}, **settings)
    first = _read(path)
    dimensions = options["classes"] - int(objective == "MultiClass")
    assert first["optimization_predictions"].shape == (dimensions, len(targets))
    assert first["optimization_predictions"].dtype == np.float32
    np.testing.assert_array_equal(first["optimization_predictions"], partial_native_state)
    if objective == "MultiClass":
        # Public raw values include the last-class gauge. Float32 subtraction
        # cannot recover all exact optimizer bits from these rounded outputs.
        reconstructed = (partial.predictions[:, :-1] - partial.predictions[:, -1:]).T
        assert not np.array_equal(reconstructed, first["optimization_predictions"])

    resumed = run_training(bins, targets, features, borders, snapshot_file=path, **options, **settings)
    complete = run_training(bins, targets, features, borders, snapshot_file=complete_path, **options, **settings)
    _assert_equal(resumed, expected)
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == 3
    final, uninterrupted = _read(path), _read(complete_path)
    np.testing.assert_array_equal(final["optimization_predictions"], final_native_state)
    for name in ("optimization_predictions", "predictions", "eval_predictions", "leaf_values", "rmse"):
        np.testing.assert_array_equal(final[name], uninterrupted[name], err_msg=name)


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("corruption", ["missing", "shape", "transposed", "dtype", "nan", "inf", "pickle"])
def test_optimizer_snapshot_corruption_fails_before_gpu(
        metal_device, monkeypatch, tmp_path, permutations, objective, corruption):
    bins, targets, features, borders, matrices, weights = _problem(permutations, rows=67)
    path = tmp_path / "invalid.snapshot"
    options = _options(objective)
    common = dict(sample_weight=weights, permutation_bins=matrices,
                  save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read(path)
    name = "permutation_optimization_predictions" if permutations > 1 else "optimization_predictions"
    dimensions = options["classes"] - int(objective == "MultiClass")
    shape = (permutations, dimensions, len(targets)) if permutations > 1 else (dimensions, len(targets))
    assert arrays[name].shape == shape
    assert arrays[name].dtype == np.float32
    if corruption == "missing":
        arrays.pop(name)
    elif corruption == "shape":
        arrays[name] = arrays[name][..., :-1]
    elif corruption == "transposed":
        arrays[name] = arrays[name].swapaxes(-1, -2)
    elif corruption == "dtype":
        arrays[name] = arrays[name].astype(np.float64)
    elif corruption == "nan":
        arrays[name].flat[0] = np.nan
    elif corruption == "inf":
        arrays[name].flat[0] = np.inf
    elif corruption == "pickle":
        arrays[name] = arrays[name].astype(object)
    _write(path, arrays)
    _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError, match="resume=False" if corruption == "missing" else None):
        run_training(bins, targets, features, borders, **options, **common)


@pytest.mark.parametrize("permutations", [1, 4])
def test_resume_false_replaces_legacy_snapshot_missing_exact_state(metal_device, tmp_path, permutations):
    bins, targets, features, borders, matrices, weights = _problem(permutations, rows=67)
    path = tmp_path / "legacy.snapshot"
    options = _options(iterations=4)
    common = dict(sample_weight=weights, permutation_bins=matrices,
                  save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    arrays = _read(path)
    name = "permutation_optimization_predictions" if permutations > 1 else "optimization_predictions"
    arrays.pop(name)
    _write(path, arrays)
    restarted = run_training(bins, targets, features, borders, resume=False, **options, **common)
    expected = run_training(bins, targets, features, borders, **options,
                            **{**common, "save_snapshot": False})
    _assert_equal(restarted, expected)
    assert restarted.stats["resumed_iterations"] == 0
    assert len(restarted.depths) == options["iterations"]
    assert name in _read(path)


@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_already_complete_snapshot_preserves_optimizer_arrays_without_gpu(
        metal_device, monkeypatch, tmp_path, permutations, objective):
    bins, targets, features, borders, matrices, weights = _problem(permutations, rows=67)
    path = tmp_path / "completed.snapshot"
    options = _options(objective, iterations=3)
    common = dict(sample_weight=weights, permutation_bins=matrices,
                  save_snapshot=True, snapshot_file=path)
    expected = run_training(bins, targets, features, borders, **options, **common)
    before = _read(path)
    _forbid_gpu(monkeypatch)
    actual = run_training(bins, targets, features, borders, **options, **common)
    _assert_equal(actual, expected)
    assert actual.stats["resumed_iterations"] == options["iterations"]
    after = _read(path)
    assert set(after) == set(before)
    for name in before:
        if name != "metadata":
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
