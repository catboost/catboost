"""Multi-output lifecycle checks on Metal with independent metric equations."""

import platform

import numpy as np
import pytest

from catboost_metal import _multioutput, _multiclass, _native
from catboost_metal._data import cuda_search_permutation
from catboost_metal._training import metric, run_training


OBJECTIVES = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

    def forbidden(*args, **kwargs):
        pytest.fail("Multi-output lifecycle tests must not train CatBoost on CPU")

    for cls in (CatBoost, CatBoostClassifier, CatBoostRegressor):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture(scope="module")
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Multi-output lifecycle tests require Apple Silicon Metal")


def _problem(objective, permutations=1, rows=129):
    rng = np.random.default_rng(41693)
    bins = rng.integers(0, 8, size=(3, rows), dtype=np.uint8)
    signal = np.column_stack(((bins[0].astype(float) - 3) / 4,
                              (bins[1].astype(float) - 4) / 5,
                              (bins[2].astype(float) - 3) / 4)).astype(np.float32)
    if objective == "MultiRMSE":
        targets = signal
    elif objective == "RMSEWithUncertainty":
        targets = (signal[:, 0] + rng.normal(0, .2, rows)).astype(np.float32)
    elif objective == "MultiLogloss":
        targets = (signal > 0).astype(np.float32)
    else:
        targets = (1 / (1 + np.exp(-signal))).astype(np.float32)
    banks = [bins.copy()]
    for _ in range(permutations - 1):
        current = bins.copy()
        for column in current:
            rng.shuffle(column)
        banks.append(current)
    weights = rng.uniform(.2, 2., rows).astype(np.float32)
    weights[::19] = 0
    features = np.repeat(np.arange(3, dtype=np.uint32), 7)
    borders = np.tile(np.arange(7, dtype=np.uint32), 3)
    return bins, targets, features, borders, tuple(banks), weights


def _options(objective, **overrides):
    dimensions = 2 if objective == "RMSEWithUncertainty" else 3
    result = dict(objective=objective, classes=dimensions, iterations=7, depth=2,
                  learning_rate=.13, l2_leaf_reg=3., score_function="L2",
                  bias=np.array([.11, -.07, .19][:dimensions], np.float32),
                  leaf_estimation_iterations=2, random_seed=671, iteration_offset=17,
                  random_strength=.3)
    result.update(overrides)
    return result


def _oracle(raw, target, weights, objective):
    raw, target = np.asarray(raw, np.float64), np.asarray(target, np.float64)
    if objective == "MultiRMSE":
        per_row = np.sum((raw - target) ** 2, axis=1)
    elif objective == "RMSEWithUncertainty":
        residual = target - raw[:, 0]
        per_row = .5 * np.log(2 * np.pi) + raw[:, 1] + .5 * residual**2 * np.exp(np.minimum(-2 * raw[:, 1], 70))
    else:
        per_row = np.mean(np.maximum(raw, 0) - target * raw + np.log1p(np.exp(-np.abs(raw))), axis=1)
    value = np.average(per_row, weights=weights)
    return np.sqrt(value) if objective == "MultiRMSE" else value


def _assert_equal(actual, expected):
    for name in ("depths", "split_features", "split_bins", "split_types",
                 "leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name), err_msg=name)


def _read(path):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    assert all(array.dtype.kind != "O" for array in arrays.values())
    return arrays


def _forbid_gpu(monkeypatch):
    from catboost_metal import _evaluation

    for module in (_native, _multiclass, _multioutput):
        monkeypatch.setattr(module, "Session", lambda *a, **k: pytest.fail("Invalid targets opened a GPU session"))
        monkeypatch.setattr(module, "train", lambda *a, **k: pytest.fail("Invalid targets started GPU training"))
    monkeypatch.setattr(_evaluation, "EvaluationCursor",
                        lambda *a, **k: pytest.fail("Invalid targets opened GPU validation"))


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("weighted", [False, True])
def test_vector_metrics_match_independent_stable_formulas(objective, weighted):
    weights = np.array([.1, 3, 0, .75]) if weighted else None
    if objective == "RMSEWithUncertainty":
        raw = np.array([[.5, -50], [1.25, 0], [-.7, 1], [0, -4]])
        target = np.array([.500001, 2, -.6, 0])
    elif objective == "MultiRMSE":
        raw = np.array([[1, -2, 3], [4, 5, -6], [7, 8, 9], [0, .5, -1]])
        target = np.array([[.5, 1, 2], [3, -1, 0], [7, 2, 8], [1, -1, 1]])
    else:
        raw = np.array([[-1000, 1000, 0], [1000, -1000, .5], [3, -5, 7], [-1, 2, -.5]])
        target = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1], [1, 0, 1]], float)
        if objective == "MultiCrossEntropy":
            target = .8 * target + .1
    assert metric(raw, target, weights, objective) == pytest.approx(
        _oracle(raw, target, weights, objective), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("permutations", [1, 4])
@pytest.mark.parametrize("sampling", [
    pytest.param({"bootstrap_type": "No"}, id="No"),
    pytest.param({"bootstrap_type": "Bayesian", "bagging_temperature": 1.5}, id="Bayesian"),
    pytest.param({"bootstrap_type": "Bernoulli", "subsample": .6}, id="Bernoulli"),
    pytest.param({"bootstrap_type": "Poisson", "subsample": .7}, id="Poisson"),
])
def test_multioutput_all_sampler_snapshots_restore_exactly(metal_device, tmp_path, objective, permutations, sampling):
    bins, targets, features, borders, banks, weights = _problem(objective, permutations)
    options = _options(objective, **sampling)
    with _multioutput.Session(bins, targets, features, borders, sample_weight=weights, **options) as native:
        if permutations > 1:
            native.configure_permutations(banks)
        for iteration in range(options["iterations"]):
            if permutations > 1:
                native.select_permutation(cuda_search_permutation(options["random_seed"],
                    options["iteration_offset"] + iteration, permutations))
            native.step()
        expected = native.result()
        optimizer = native.optimization_predictions()
        permutation_state = native.permutation_state if permutations > 1 else None
    path = tmp_path / "vector.snapshot"
    common = dict(permutation_bins=banks, sample_weight=weights, eval_bins=banks[-1],
                  eval_targets=targets, eval_weight=weights, use_best_model=False,
                  save_snapshot=True, snapshot_file=path, snapshot_interval=0)
    first = run_training(bins, targets, features, borders, **{**options, "iterations": 2}, **common)
    first_stats = first.stats["validation"]
    assert first_stats["dataset_uploads"] == 1
    assert first_stats["bins_upload_bytes"] == bins.nbytes
    assert first_stats["output_dimensions"] == options["classes"]
    assert first_stats["kernel_dispatches"] == 2
    resumed = run_training(bins, targets, features, borders, **options, **common)
    complete = run_training(bins, targets, features, borders, **options,
                            **{**common, "save_snapshot": False})
    _assert_equal(resumed, expected)
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == 2
    assert resumed.stats["validation"]["dataset_uploads"] == 2
    assert resumed.stats["validation"]["kernel_dispatches"] == 7
    assert resumed.predictions.shape == (len(targets), options["classes"])
    arrays = _read(path)
    if permutations > 1:
        for name, value in permutation_state.items():
            np.testing.assert_array_equal(arrays["permutation_" + name], value, err_msg=name)
        assert arrays["permutation_optimization_predictions"].shape == (permutations, options["classes"], len(targets))
    else:
        np.testing.assert_array_equal(arrays["optimization_predictions"], optimizer)
        assert arrays["optimization_predictions"].shape == (options["classes"], len(targets))
    assert resumed.evals_result["validation"][objective][-1] == pytest.approx(
        _oracle(arrays["eval_predictions"], targets, weights, objective), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("objective", OBJECTIVES)
@pytest.mark.parametrize("vector_bias", [False, True])
def test_dimensions_are_inferred_with_scalar_or_vector_bias(metal_device, objective, vector_bias):
    bins, targets, features, borders, _, weights = _problem(objective, rows=67)
    options = _options(objective, iterations=3)
    dimensions = options.pop("classes")
    if not vector_bias:
        options["bias"] = .125
    actual = run_training(bins, targets, features, borders, sample_weight=weights,
                          eval_bins=bins, eval_targets=targets, use_best_model=False, **options)
    expected = _multioutput.train(bins, targets, features, borders, sample_weight=weights, **options)
    _assert_equal(actual, expected)
    assert actual.predictions.shape == (len(targets), dimensions)
    assert actual.stats["validation"]["output_dimensions"] == dimensions


@pytest.mark.parametrize("permutations", [1, 4])
def test_multirmse_early_stopping_trims_shared_vector_tree(metal_device, tmp_path, permutations):
    bins = np.zeros((1, 67), np.uint8)
    targets = np.tile(np.array([1., -2., .5], np.float32), (67, 1))
    empty = np.empty(0, np.uint32)
    path = tmp_path / "best.vector.snapshot"
    options = _options("MultiRMSE", depth=0, learning_rate=.5, l2_leaf_reg=0,
                       bias=0., random_strength=0, leaf_estimation_iterations=1)
    result = run_training(bins, targets, empty, empty,
        permutation_bins=tuple(bins.copy() for _ in range(permutations)),
        eval_bins=bins, eval_targets=-targets, early_stopping_rounds=2, use_best_model=True,
        save_snapshot=True, snapshot_file=path, **options)
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert len(result.depths) == 1
    assert result.leaf_values.shape == (1, 1, 3)
    np.testing.assert_array_equal(result.predictions, targets * .5)
    np.testing.assert_allclose(result.evals_result["validation"]["MultiRMSE"],
                               np.sqrt(5.25) * np.array([1.5, 1.75, 1.875]), rtol=1e-12)
    arrays = _read(path)
    assert len(arrays["depths"]) == 3
    np.testing.assert_array_equal(arrays["predictions"], targets * .875)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_custom_unweighted_vector_metric_preserves_weighted_objective_history(metal_device, tmp_path, objective):
    bins, targets, features, borders, _, weights = _problem(objective, rows=67)
    path = tmp_path / "custom.metric.snapshot"
    selection = objective + ":use_weights=false"
    result = run_training(bins, targets, features, borders, sample_weight=weights,
        eval_bins=bins, eval_targets=targets, eval_weight=weights, use_best_model=False,
        eval_metric=selection, save_snapshot=True, snapshot_file=path,
        **_options(objective, iterations=3))
    raw = _read(path)["eval_predictions"]
    assert result.evals_result["validation"][selection][-1] == pytest.approx(
        _oracle(raw, targets, None, objective), rel=1e-12, abs=1e-12)
    assert result.evals_result["validation"][objective][-1] == pytest.approx(
        _oracle(raw, targets, weights, objective), rel=1e-12, abs=1e-12)
    assert result.stats["selection_metric"] == selection
    assert result.stats["metric_maximized"] is False


def test_multilabel_accuracy_maximization_controls_best_model(metal_device):
    bins = np.zeros((1, 67), np.uint8)
    targets = np.ones((67, 3), np.float32)
    empty = np.empty(0, np.uint32)
    options = _options("MultiLogloss", depth=0, learning_rate=.15, bias=-1.,
                       random_strength=0., leaf_estimation_iterations=1)
    result = run_training(bins, targets, empty, empty, eval_bins=bins, eval_targets=targets,
                          eval_metric="Accuracy", early_stopping_rounds=2, use_best_model=True, **options)
    values = result.evals_result["validation"]["Accuracy"]
    assert values[0] == 0 and values[-1] == 1
    assert result.best_iteration == int(np.argmax(values)) > 0
    assert result.best_score["validation"]["Accuracy"] == 1
    assert result.stats["metric_maximized"] is True
    assert result.stats["stop_reason"] == "early_stopping"
    assert len(result.depths) == result.best_iteration + 1
    one = run_training(bins, targets, empty, empty,
                       **{**options, "iterations": result.best_iteration + 1})
    np.testing.assert_allclose(result.predictions, one.predictions, rtol=1e-6, atol=1e-6)


def test_negative_uncertainty_nll_remains_valid_across_snapshot_resume(metal_device, tmp_path):
    bins = np.zeros((1, 67), np.uint8)
    targets = np.linspace(-.001, .001, 67, dtype=np.float32)
    empty = np.empty(0, np.uint32)
    path = tmp_path / "negative.nll.snapshot"
    options = _options("RMSEWithUncertainty", depth=0, learning_rate=.01,
                       bias=np.array([0., -3.], np.float32), random_strength=0.,
                       leaf_estimation_method="Gradient", leaf_estimation_iterations=1)
    common = dict(eval_bins=bins, eval_targets=targets, use_best_model=False,
                  save_snapshot=True, snapshot_file=path)
    first = run_training(bins, targets, empty, empty, **{**options, "iterations": 2}, **common)
    assert (first.rmse < 0).all()
    assert (_read(path)["rmse"] < 0).all()
    assert all(value < 0 for value in first.evals_result["validation"]["RMSEWithUncertainty"])
    resumed = run_training(bins, targets, empty, empty, **options, **common)
    complete = run_training(bins, targets, empty, empty, **options,
                            **{**common, "save_snapshot": False})
    _assert_equal(resumed, complete)
    assert resumed.evals_result == complete.evals_result
    assert (resumed.rmse < 0).all()
    assert resumed.stats["resumed_iterations"] == 2


@pytest.mark.parametrize("objective,change", [
    ("MultiRMSE", "vector_target"), ("MultiRMSE", "one_column"),
    ("MultiRMSE", "wrong_classes"), ("MultiRMSE", "nan_target"),
    ("MultiRMSE", "wrong_eval_dimensions"), ("MultiRMSE", "too_many_dimensions"),
    ("MultiRMSE", "invalid_classes"), ("MultiLogloss", "fractional_target"),
    ("MultiLogloss", "fractional_eval_target"), ("MultiCrossEntropy", "outside_probability"),
    ("MultiCrossEntropy", "wrong_rows"), ("RMSEWithUncertainty", "matrix_target"),
    ("RMSEWithUncertainty", "one_column_target"), ("RMSEWithUncertainty", "wrong_classes"),
    ("RMSEWithUncertainty", "matrix_eval_target"), ("MultiLogloss", "complex_target"),
])
def test_invalid_vector_targets_and_dimensions_fail_before_gpu(monkeypatch, objective, change):
    bins, targets, features, borders, _, weights = _problem(objective, rows=67)
    options = _options(objective)
    eval_targets = targets.copy()
    if change == "vector_target":
        targets = targets[:, 0]
    elif change == "one_column":
        targets = targets[:, :1]
        options.pop("classes")
    elif change == "wrong_classes":
        options["classes"] = 3 if objective == "RMSEWithUncertainty" else 2
    elif change == "nan_target":
        targets.flat[0] = np.nan
    elif change == "wrong_eval_dimensions":
        eval_targets = eval_targets[:, :2]
    elif change == "too_many_dimensions":
        targets = np.zeros((len(targets), 65), np.float32)
        options.pop("classes")
    elif change == "invalid_classes":
        options["classes"] = True
    elif change == "fractional_target":
        targets.flat[0] = .3
    elif change == "fractional_eval_target":
        eval_targets.flat[0] = .3
    elif change == "outside_probability":
        targets.flat[0] = 1.3
    elif change == "wrong_rows":
        targets = targets[:-1]
    elif change == "matrix_target":
        targets = np.column_stack((targets, targets))
    elif change == "one_column_target":
        targets = targets[:, None]
    elif change == "matrix_eval_target":
        eval_targets = np.column_stack((eval_targets, eval_targets))
    elif change == "complex_target":
        targets = targets.astype(np.complex64)
    _forbid_gpu(monkeypatch)
    with pytest.raises(ValueError):
        run_training(bins, targets, features, borders, sample_weight=weights,
                     eval_bins=bins, eval_targets=eval_targets, **options)
