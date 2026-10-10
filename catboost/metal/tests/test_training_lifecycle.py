"""Validation, early stopping, and safe continuation using the real Metal GPU."""

import json
import platform

import numpy as np
import pytest

from catboost_metal import _native
from catboost_metal._training import metric, run_training, _read_snapshot, _write_snapshot, _shared_metric


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    from catboost import CatBoost, CatBoostRegressor, CatBoostClassifier

    def forbidden(*args, **kwargs):
        raise AssertionError("Lifecycle tests must not call CatBoost CPU training")

    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)


@pytest.fixture
def metal_device():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Metal checks require Apple Silicon macOS")
    assert _native.device_info()["backend"] == "Metal"


def _problem():
    bins = np.asarray([[0, 0, 1, 1, 0, 1]], dtype=np.uint8)
    targets = np.where(bins[0] > 0, 1.0, -1.0).astype(np.float32)
    return bins, targets, np.asarray([0], np.uint32), np.asarray([0], np.uint32)


def _params(**overrides):
    values = dict(iterations=8, depth=2, learning_rate=0.5, l2_leaf_reg=0,
                  bias=0.0, score_function="L2", objective="RMSE")
    values.update(overrides)
    return values


def test_weighted_metrics_are_stable_at_extreme_logits():
    assert metric([-1000, 1000], [0, 1], [2, 3], "Logloss") == 0.0
    assert metric([-1000, 1000], [1, 0], [2, 3], "Logloss") == 1000.0
    assert metric([0, 2], [0, 0], [3, 1], "RMSE") == 1.0
    np.testing.assert_allclose(metric([0, 0], [0.2, 0.8], objective="CrossEntropy"), np.log(2))


def test_metric_weight_flags_and_registry_defaults_survive_shared_utility_override():
    assert _shared_metric("RMSE:use_weights=false", [0, 0], [0, 1], [1, 9]) == pytest.approx(np.sqrt(0.5))
    assert _shared_metric("Accuracy:use_weights=false", [0, 0], [0, 1], [1, 9]) == 0.5
    assert _shared_metric("Accuracy:use_weights=true", [0, 0], [0, 1], [1, 9]) == 0.1
    raw, targets, weights = [3, 1, 2, 0], [1, 1, 0, 0], [1, 9, 1, 1]
    assert _shared_metric("AUC", raw, targets, weights) == 0.75
    assert _shared_metric("AUC:use_weights=false", raw, targets, weights) == 0.75
    assert _shared_metric("AUC:use_weights=true", raw, targets, weights) == pytest.approx(0.55)


def test_cuda_mae_and_quantile_metrics_preserve_sub_delta_residuals():
    from catboost.utils import eval_metric

    raw, targets, weights = np.zeros(2), np.asarray([1e-7, -2e-7]), np.asarray([1, 3])
    assert eval_metric(targets, raw, "MAE", weight=weights, thread_count=1)[0] == 0
    assert _shared_metric("MAE", raw, targets, weights) == pytest.approx(1.75e-7)
    assert _shared_metric("MAE:use_weights=false", raw, targets, weights) == pytest.approx(1.5e-7)
    assert _shared_metric("Quantile:alpha=0.2;delta=0.001", raw, targets, weights) == pytest.approx(1.25e-7)
    assert _shared_metric("Quantile:alpha=0.2;delta=0.001;use_weights=false", raw, targets, weights) == pytest.approx(0.9e-7)


def test_multiclass_metrics_keep_small_terms_under_large_common_logits():
    assert metric([[1e30, 1e30, 1e30]], [0], objective="MultiClass") == pytest.approx(np.log(3))
    assert metric([[1e30, 0, 0]], [0], objective="MultiClassOneVsAll") == pytest.approx(2 * np.log(2) / 3)
    raw = [[1000, -1000, -1000], [1000, -1000, -1000]]
    assert metric(raw, [0, 1], [1, 3], "MultiClass") == 1500
    assert metric(raw, [0, 1], [1, 3], "MultiClassOneVsAll") == 500


@pytest.mark.parametrize("corruption", ["pickle", "feature_index", "weights", "history",
                                         "best_iteration", "bootstrap_offset", "mvs_lambda"])
def test_corrupt_snapshots_are_rejected_without_deserializing_objects(tmp_path, corruption):
    path = tmp_path / "snapshot.npz"
    result = _native.TrainResult(
        depths=np.asarray([1], np.uint32), split_features=np.asarray([[0]], np.uint32),
        split_bins=np.asarray([[0]], np.uint32), leaf_values=np.asarray([[-1, 1]], np.float32),
        leaf_weights=np.asarray([[1, 1]], np.float32), predictions=np.asarray([-1, 1], np.float32),
        rmse=np.asarray([1, 0], np.float32), stats={"device": "fixture", "gpu_seconds": 0,
                                                "kernel_dispatches": 1})
    result.split_types = np.zeros((1, 1), np.uint8)
    _write_snapshot(path, "expected", result, {"learn": {"RMSE": [0]}}, -1, np.inf, None)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if corruption == "pickle":
        arrays["predictions"] = np.asarray([object(), object()], dtype=object)
    elif corruption == "feature_index":
        arrays["split_features"][0, 0] = 1
    elif corruption == "weights":
        arrays["leaf_weights"][0, 0] = -1
    else:
        header = json.loads(str(arrays["metadata"].item()))
        if corruption == "history":
            header["history"]["learn"]["RMSE"] = [0, 1]
        elif corruption == "best_iteration":
            header["best_iteration"] = 0
        else:
            header["stats"]["bootstrap_state"] = {
                "iteration_offset": 2 if corruption == "bootstrap_offset" else 1,
                "mvs_lambda": -1 if corruption == "mvs_lambda" else 0}
        arrays["metadata"] = np.asarray(json.dumps(header))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    with pytest.raises(ValueError):
        _read_snapshot(path, "expected", rows=2, features=1, depth=1,
                       iterations=2, objective="RMSE", eval_rows=None)


@pytest.mark.parametrize("options", [
    {"early_stopping_rounds": 0}, {"early_stopping_rounds": True},
    {"early_stopping_rounds": 2}, {"use_best_model": True},
    {"snapshot_interval": -1}, {"snapshot_interval": np.inf},
    {"save_snapshot": 1}, {"resume": 1}, {"callback": 4},
    {"eval_targets": [1]}, {"objective": "NotAnObjective"},
    {"eval_metric": ""}, {"eval_metric": 2}, {"eval_metric": "UnknownMetric"},
    {"eval_metric": "RMSE:unknown=1"}, {"eval_metric": "QueryRMSE"},
])
def test_invalid_lifecycle_options_fail_before_gpu(monkeypatch, options):
    monkeypatch.setattr(_native, "train", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Unexpected GPU call"), raising=False)
    with pytest.raises((ValueError, TypeError)):
        run_training(*_problem(), **_params(**options))


def test_early_stopping_retains_best_tree_and_matching_predictions(metal_device):
    bins, targets, features, borders = _problem()
    result = run_training(bins, targets, features, borders,
                          eval_bins=bins, eval_targets=-targets,
                          early_stopping_rounds=2, use_best_model=True, **_params())
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert result.stats["stop_reason"] == "early_stopping"
    assert result.stats["iterations_trained"] == 3
    assert len(result.depths) == 1
    assert len(result.rmse) == 4
    assert len(result.evals_result["validation"]["RMSE"]) == 3
    np.testing.assert_allclose(result.predictions, targets * 0.5, atol=1e-6)
    assert result.best_score["validation"]["RMSE"] == pytest.approx(1.5)


def test_early_stopping_without_trimming_retains_completed_trees(metal_device):
    bins, targets, features, borders = _problem()
    result = run_training(bins, targets, features, borders,
                          eval_bins=bins, eval_targets=-targets,
                          early_stopping_rounds=2, use_best_model=False, **_params())
    assert len(result.depths) == 3
    np.testing.assert_allclose(result.predictions, targets * 0.875, atol=1e-6)


@pytest.mark.parametrize("objective", ["RMSE", "Logloss", "CrossEntropy"])
def test_snapshot_resume_matches_uninterrupted_gpu_training(metal_device, tmp_path, objective):
    bins, targets, features, borders = _problem()
    if objective != "RMSE":
        targets = (targets + 1) / 2
        if objective == "CrossEntropy":
            targets = targets * 0.8 + 0.1
    weights = np.asarray([1, 2, 1, 2, 0, 3], np.float32)
    common = dict(objective=objective, sample_weight=weights, eval_bins=bins,
                  eval_targets=targets, eval_weight=weights, use_best_model=False,
                  metadata={"borders": [[0.5]], "features": ["x"]})
    path = tmp_path / "training.snapshot"
    first = run_training(bins, targets, features, borders, save_snapshot=True,
                         snapshot_file=path, **_params(iterations=3, **common))
    assert len(first.depths) == 3
    resumed = run_training(bins, targets, features, borders, save_snapshot=True,
                           snapshot_file=path, **_params(**common))
    uninterrupted = run_training(bins, targets, features, borders, **_params(**common))
    assert resumed.stats["resumed_iterations"] == 3
    for name in ("depths", "split_features", "split_bins", "split_types"):
        np.testing.assert_array_equal(getattr(resumed, name), getattr(uninterrupted, name))
    for name in ("leaf_values", "leaf_weights", "predictions", "rmse"):
        np.testing.assert_allclose(getattr(resumed, name), getattr(uninterrupted, name), atol=2e-6)
    np.testing.assert_allclose(resumed.evals_result["validation"][objective],
                               uninterrupted.evals_result["validation"][objective], atol=2e-6)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        header = json.loads(str(archive["metadata"].item()))
        assert header["completed_iterations"] == 8
        np.testing.assert_array_equal(archive["predictions"], resumed.predictions)


def test_snapshot_binds_data_and_preprocessing_before_gpu(metal_device, monkeypatch, tmp_path):
    args = _problem()
    path = tmp_path / "training.snapshot"
    run_training(*args, save_snapshot=True, snapshot_file=path,
                 metadata={"borders": [[0.5]]}, **_params(iterations=2))
    monkeypatch.setattr(_native, "Session", lambda *a, **k: pytest.fail("Unexpected GPU call"))
    for changed_args, metadata in ((args, {"borders": [[0.7]]}),
                                   ((args[0], args[1] + 0.1, *args[2:]), {"borders": [[0.5]]})):
        with pytest.raises(ValueError, match="does not match"):
            run_training(*changed_args, save_snapshot=True, snapshot_file=path,
                         metadata=metadata, **_params())


def test_snapshot_keeps_untrimmed_state_after_early_stopping(metal_device, tmp_path):
    bins, targets, features, borders = _problem()
    path = tmp_path / "training.snapshot"
    options = _params(eval_bins=bins, eval_targets=-targets, early_stopping_rounds=2,
                      use_best_model=True, save_snapshot=True, snapshot_file=path)
    result = run_training(bins, targets, features, borders, **options)
    assert len(result.depths) == 1
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive["depths"]) == 3
        np.testing.assert_allclose(archive["predictions"], targets * 0.875, atol=1e-6)
    resumed = run_training(bins, targets, features, borders, **options)
    assert resumed.stats["resumed_iterations"] == 3
    assert resumed.stats["iterations_trained"] == 3
    np.testing.assert_array_equal(resumed.predictions, result.predictions)


def test_callback_can_stop_and_resume_without_mutating_history(metal_device, tmp_path):
    path = tmp_path / "training.snapshot"

    def stop(info):
        assert info.iteration == 1
        info.metrics["learn"]["RMSE"].append(-100)
        return False

    stopped = run_training(*_problem(), callback=stop, save_snapshot=True,
                           snapshot_file=path, **_params())
    assert len(stopped.depths) == 1
    assert stopped.stats["stop_reason"] == "callback"
    assert len(stopped.evals_result["learn"]["RMSE"]) == 1
    resumed = run_training(*_problem(), save_snapshot=True,
                           snapshot_file=path, **_params())
    assert len(resumed.depths) == 8
    assert resumed.stats["resumed_iterations"] == 1


def test_public_fit_exposes_best_model_and_resumes_prepared_feature_state(metal_device, tmp_path):
    from catboost_metal import CatBoostMetalRegressor

    bins, targets, _, _ = _problem()
    features = bins.T.astype(np.float32)
    options = dict(iterations=8, depth=2, learning_rate=0.5, l2_leaf_reg=0,
                   border_count=1, boost_from_average=False, score_function="L2")
    best = CatBoostMetalRegressor(**options).fit(
        features, targets, eval_set=(features, -targets),
        early_stopping_rounds=2, use_best_model=True)
    assert best.get_best_iteration() == 0
    assert best.tree_count_ == 1
    assert len(best.get_evals_result()["validation"]["RMSE"]) == 3
    np.testing.assert_allclose(best.predict(features, task_type="METAL"),
                               best.training_predictions_, atol=1e-6)
    np.testing.assert_allclose(best.predict(features), best.training_predictions_, atol=1e-6)
    path = tmp_path / "snapshot.npz"
    CatBoostMetalRegressor(**{**options, "iterations": 3}).fit(
        features, targets, save_snapshot=True, snapshot_file=path)
    resumed = CatBoostMetalRegressor(**options).fit(
        features, targets, save_snapshot=True, snapshot_file=path)
    uninterrupted = CatBoostMetalRegressor(**options).fit(features, targets)
    np.testing.assert_array_equal(resumed.training_predictions_, uninterrupted.training_predictions_)
    assert resumed.training_stats_["resumed_iterations"] == 3


@pytest.mark.parametrize("selection,maximize", [("R2", True), ("MAE", False)])
def test_shared_selection_metric_uses_correct_best_direction(metal_device, selection, maximize):
    bins, targets, features, borders = _problem()
    result = run_training(bins, targets, features, borders, eval_bins=bins,
                          eval_targets=-targets, eval_metric=selection,
                          early_stopping_rounds=2, use_best_model=True, **_params())
    history = result.evals_result["validation"][selection]
    assert (history[0] > history[-1]) == maximize
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert len(result.depths) == 1
    assert result.best_score["validation"][selection] == history[0]
    assert set(result.evals_result["validation"]) == {"RMSE", selection}
    assert result.stats["metric_maximized"] is maximize


def test_parameterized_accuracy_selects_later_tree_instead_of_objective_minimum(metal_device):
    bins, targets, features, borders = _problem()
    selected = "Accuracy:proba_border=0.7"
    result = run_training(bins, targets, features, borders, eval_bins=bins,
                          eval_targets=(targets + 1) / 2, eval_metric=selected,
                          early_stopping_rounds=2, use_best_model=True, **_params())
    assert result.evals_result["validation"][selected] == [0.5, 0.5, 1.0, 1.0, 1.0]
    assert result.best_iteration == 2
    assert result.stopped_iteration == 4
    assert len(result.depths) == 3
    assert result.best_score["validation"][selected] == 1.0
    np.testing.assert_allclose(result.predictions, targets * 0.875, atol=1e-6)


@pytest.mark.parametrize("selection", ["MAE:use_weights=false", "R2:use_weights=true", "AUC:use_weights=true"])
def test_parameterized_metrics_and_weights_match_shared_catboost_utility(metal_device, selection):
    from catboost.utils import eval_metric

    bins, targets, features, borders = _problem()
    validation = ((targets + 1) / 2 if selection.startswith("AUC") else
                  np.asarray([-2, -0.25, 3, 1, -3, 2], np.float32))
    weights = np.asarray([1, 9, 1, 20, 0, 4], np.float32)
    result = run_training(bins, targets, features, borders, eval_bins=bins,
                          eval_targets=validation, eval_weight=weights, eval_metric=selection,
                          use_best_model=False, **_params(iterations=4))
    expected = []
    for tree in range(4):
        raw = targets * (1 - 0.5 ** (tree + 1))
        if selection == "MAE:use_weights=false":
            expected.append(float(np.mean(np.abs(validation.astype(np.float64) - raw))))
        else:
            expected.append(eval_metric(validation, raw, selection, weight=weights, thread_count=1)[0])
    np.testing.assert_allclose(result.evals_result["validation"][selection], expected, atol=1e-7)
    assert result.best_iteration == int(np.argmax(expected) if selection.startswith(("AUC", "R2"))
                                        else np.argmin(expected))


def test_maximizing_metric_snapshot_preserves_best_state_and_full_history(metal_device, tmp_path):
    bins, targets, features, borders = _problem()
    path = tmp_path / "maximizing.snapshot"
    options = dict(eval_bins=bins, eval_targets=targets * 0.5, eval_metric="R2", use_best_model=False)
    run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path,
                 **_params(iterations=3, **options))
    resumed = run_training(bins, targets, features, borders, save_snapshot=True,
                           snapshot_file=path, **_params(**options))
    complete = run_training(bins, targets, features, borders, **_params(**options))
    assert resumed.best_iteration == complete.best_iteration == 0
    assert resumed.best_score["validation"]["R2"] == 1.0
    assert resumed.evals_result == complete.evals_result
    assert resumed.stats["resumed_iterations"] == 3
    with pytest.raises(ValueError, match="does not match"):
        run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path,
                     **_params(**{**options, "eval_metric": "MAE"}))


def test_selection_metric_without_validation_reports_learn_history(metal_device):
    result = run_training(*_problem(), eval_metric="R2", **_params(iterations=3))
    assert result.best_iteration == -1
    assert set(result.evals_result["learn"]) == {"RMSE", "R2"}
    np.testing.assert_allclose(result.evals_result["learn"]["R2"], [0.75, 0.9375, 0.984375])
    assert result.best_score["learn"]["R2"] == 0.984375


def test_validation_keeps_dataset_resident_across_trees_and_reports_uploads(metal_device, monkeypatch, tmp_path):
    from catboost_metal import _inference

    monkeypatch.setattr(_inference, "predict_bins", lambda *a, **kw: pytest.fail("Stateless validation inference used"))
    bins, targets, features, borders = _problem()
    path = tmp_path / "resident.snapshot"
    options = dict(eval_bins=bins, eval_targets=targets, use_best_model=False,
                   save_snapshot=True, snapshot_file=path, snapshot_interval=0)
    first = run_training(bins, targets, features, borders, **_params(iterations=3, **options))
    stats = first.stats["validation"]
    assert stats["dataset_uploads"] == 1
    assert stats["bins_upload_bytes"] == bins.nbytes
    assert stats["kernel_dispatches"] == 3
    assert stats["resident_bytes"] >= bins.nbytes + len(targets) * 4
    resumed = run_training(bins, targets, features, borders, **_params(**options))
    stats = resumed.stats["validation"]
    assert stats["dataset_uploads"] == 2
    assert stats["bins_upload_bytes"] == bins.nbytes * 2
    assert stats["kernel_dispatches"] == 8


@pytest.mark.parametrize("objective,parameter,history_name", [
    ("Poisson", 1.0, "Poisson"), ("Huber", 1.25, "Huber:delta=1.25"),
    ("Expectile", 0.3, "Expectile:alpha=0.3"),
    ("Lq", 1.5, "Lq:q=1.5"), ("Tweedie", 1.5, "Tweedie:variance_power=1.5"),
    ("LogLinQuantile", 0.3, "LogLinQuantile:alpha=0.3"),
    ("Quantile", 0.3, "Quantile:alpha=0.3"), ("MAE", 0.5, "MAE"), ("MAPE", 1.0, "MAPE"),
])
def test_extended_objective_validation_uses_full_parameterized_metric(
        metal_device, objective, parameter, history_name):
    from catboost.utils import eval_metric

    bins, targets, features, borders = _problem()
    targets = targets + 3 if objective in ("Poisson", "Tweedie") else targets * 3
    result = run_training(bins, targets, features, borders, eval_bins=bins,
                          eval_targets=targets, eval_metric="MAE", use_best_model=False,
                          objective_param=parameter, leaf_estimation_method="Gradient",
                          **_params(objective=objective, iterations=4, learning_rate=0.1, l2_leaf_reg=3))
    assert history_name in result.evals_result["learn"]
    assert set(result.evals_result["validation"]) == {history_name, "MAE"}
    if objective == "MAE":
        expected = float(np.mean(np.abs(targets.astype(np.float64) - result.predictions)))
    elif objective == "Quantile":
        residual = targets.astype(np.float64) - result.predictions
        expected = float(np.mean(np.where(residual > 0, parameter * residual, -(1 - parameter) * residual)))
    else:
        expected = eval_metric(targets, result.predictions, history_name, thread_count=1)[0]
    assert result.evals_result["validation"][history_name][-1] == pytest.approx(expected, abs=3e-6)


def test_poisson_snapshot_accepts_valid_negative_loss_history(metal_device, tmp_path):
    bins, targets, features, borders = _problem()
    targets = np.full_like(targets, 5)
    path = tmp_path / "poisson.snapshot"
    options = dict(objective="Poisson", objective_param=1.0, learning_rate=0.25,
                   l2_leaf_reg=3, eval_bins=bins, eval_targets=targets, use_best_model=False)
    partial = run_training(bins, targets, features, borders, save_snapshot=True,
                           snapshot_file=path, **_params(iterations=2, **options))
    assert (partial.rmse[1:] < 0).all()
    resumed = run_training(bins, targets, features, borders, save_snapshot=True,
                           snapshot_file=path, **_params(iterations=4, **options))
    complete = run_training(bins, targets, features, borders, **_params(iterations=4, **options))
    np.testing.assert_allclose(resumed.predictions, complete.predictions, atol=1e-6)
    assert resumed.evals_result == complete.evals_result


def test_older_no_bootstrap_snapshot_can_resume_repeatedly(metal_device, tmp_path):
    path = tmp_path / "old.snapshot"
    run_training(*_problem(), save_snapshot=True, snapshot_file=path, **_params(iterations=2))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    header = json.loads(str(arrays["metadata"].item()))
    header["stats"].pop("bootstrap_state")
    arrays["metadata"] = np.asarray(json.dumps(header))
    with path.open("wb") as output:
        np.savez(output, **arrays)
    resumed = run_training(*_problem(), save_snapshot=True, snapshot_file=path, **_params(iterations=4))
    assert resumed.stats["bootstrap_state"]["iteration_offset"] == 4
    resumed_again = run_training(*_problem(), save_snapshot=True, snapshot_file=path, **_params())
    uninterrupted = run_training(*_problem(), **_params())
    np.testing.assert_array_equal(resumed_again.predictions, uninterrupted.predictions)
    assert resumed_again.stats["bootstrap_state"]["iteration_offset"] == 8


@pytest.mark.parametrize("mode,expected", [("AnyImprovement", 0.0), ("Armijo", 100.0)])
def test_public_backtracking_routes_mode_and_preserves_snapshot_metadata(metal_device, tmp_path, capfd, mode, expected):
    from catboost import CatBoostRegressor
    from catboost_metal import CatBoostMetalRegressor

    features, targets = np.zeros((1, 1), np.float32), np.asarray([100], np.float32)
    options = dict(iterations=2, depth=2, learning_rate=1, l2_leaf_reg=0.005,
                   loss_function="Huber:delta=1", leaf_estimation_iterations=2,
                   leaf_estimation_method="Newton", leaf_estimation_backtracking=mode)
    snapshot = tmp_path / f"{mode}.snapshot"
    partial = CatBoostMetalRegressor(**options).fit(
        features, targets, eval_set=(features, targets), use_best_model=False,
        save_snapshot=True, snapshot_file=snapshot)
    np.testing.assert_array_equal(partial.predict(features, task_type="METAL"), [expected])
    assert partial.to_catboost().get_all_params()["leaf_estimation_backtracking"] == mode
    model_path = tmp_path / f"{mode}.json"
    partial.save_model(model_path, format="json")
    data = json.loads(model_path.read_text())
    params = data["model_info"]["params"]
    if isinstance(params, str):
        params = json.loads(params)
    assert params["tree_learner_options"]["leaf_estimation_backtracking"] == mode
    restored = CatBoostRegressor().load_model(str(model_path), format="json")
    np.testing.assert_array_equal(restored.predict(features), [expected])
    assert "is ignored, because it cannot be parsed" not in capfd.readouterr().err
    resumed = CatBoostMetalRegressor(**{**options, "iterations": 5}).fit(
        features, targets, eval_set=(features, targets), use_best_model=False,
        save_snapshot=True, snapshot_file=snapshot)
    uninterrupted = CatBoostMetalRegressor(**{**options, "iterations": 5}).fit(
        features, targets, eval_set=(features, targets), use_best_model=False)
    np.testing.assert_array_equal(resumed.training_predictions_, uninterrupted.training_predictions_)
    assert resumed.get_evals_result() == uninterrupted.get_evals_result()
    opposite = "Armijo" if mode == "AnyImprovement" else "AnyImprovement"
    with pytest.raises(ValueError, match="does not match"):
        CatBoostMetalRegressor(**{**options, "iterations": 5, "leaf_estimation_backtracking": opposite}).fit(
            features, targets, eval_set=(features, targets), use_best_model=False,
            save_snapshot=True, snapshot_file=snapshot)


def _multiclass_problem():
    labels = np.tile(np.arange(3, dtype=np.uint8), 11)
    bins = labels[None, :]
    features, borders = np.asarray([0, 0], np.uint32), np.asarray([0, 1], np.uint32)
    return bins, labels.astype(np.float32), features, borders


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_multiclass_validation_and_snapshot_restore_all_raw_dimensions(metal_device, tmp_path, objective):
    bins, targets, features, borders = _multiclass_problem()
    weights = np.resize(np.asarray([0.5, 2, 3, 0, 1], np.float32), len(targets))
    path = tmp_path / f"{objective}.snapshot"
    settings = dict(classes=3, objective=objective, bias=np.asarray([0.1, -0.2, 0.3], np.float32),
                    l2_leaf_reg=3, learning_rate=0.3, sample_weight=weights,
                    eval_bins=bins, eval_targets=targets, eval_weight=weights, use_best_model=False)
    run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path,
                 **_params(iterations=2, **settings))
    resumed = run_training(bins, targets, features, borders, save_snapshot=True, snapshot_file=path,
                           **_params(iterations=6, **settings))
    complete = run_training(bins, targets, features, borders, **_params(iterations=6, **settings))
    assert resumed.predictions.shape == (len(targets), 3)
    assert resumed.leaf_values.shape == (6, 4, 3)
    assert resumed.leaf_weights.shape == (6, 4)
    np.testing.assert_allclose(resumed.predictions, complete.predictions, atol=2e-6)
    np.testing.assert_allclose(resumed.leaf_values, complete.leaf_values, atol=2e-6)
    np.testing.assert_allclose(resumed.evals_result["validation"][objective],
                               complete.evals_result["validation"][objective], atol=2e-6)
    assert resumed.stats["validation"]["output_dimensions"] == 3
    assert resumed.stats["validation"]["dataset_uploads"] == 2
    assert resumed.stats["validation"]["kernel_dispatches"] == 6
    assert resumed.evals_result["validation"][objective][-1] == pytest.approx(
        metric(resumed.predictions, targets, weights, objective), abs=2e-6)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["eval_predictions"].shape == (len(targets), 3)
        assert archive["predictions"].shape == (len(targets), 3)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_multiclass_best_model_trimming_preserves_all_classes(metal_device, objective):
    bins, targets, features, borders = _multiclass_problem()
    settings = dict(classes=3, objective=objective, l2_leaf_reg=3, learning_rate=0.3,
                    eval_bins=bins, eval_targets=(targets + 1) % 3,
                    early_stopping_rounds=2, use_best_model=True)
    result = run_training(bins, targets, features, borders, **_params(**settings))
    one = run_training(bins, targets, features, borders,
                       **_params(iterations=1, classes=3, objective=objective, l2_leaf_reg=3, learning_rate=0.3))
    assert result.best_iteration == 0
    assert result.stopped_iteration == 2
    assert len(result.depths) == 1
    np.testing.assert_allclose(result.predictions, one.predictions, atol=2e-6)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
def test_public_multiclass_early_stopping_and_snapshot_roundtrip(metal_device, tmp_path, objective):
    from catboost_metal import CatBoostMetalClassifier

    bins, targets, _, _ = _multiclass_problem()
    features = bins.T.astype(np.float32)
    labels = np.asarray(["ash", "birch", "cedar"])[targets.astype(int)]
    wrong = np.asarray(["ash", "birch", "cedar"])[((targets + 1) % 3).astype(int)]
    options = dict(iterations=6, depth=2, learning_rate=0.3, l2_leaf_reg=3,
                   loss_function=objective, leaf_estimation_iterations=1)
    model = CatBoostMetalClassifier(**options).fit(
        features, labels, eval_set=(features, wrong), early_stopping_rounds=2, use_best_model=True)
    assert model.get_best_iteration() == 0 and model.tree_count_ == 1
    np.testing.assert_allclose(model.predict(features, prediction_type="RawFormulaVal", task_type="METAL"),
                               model.training_predictions_, atol=2e-6)
    assert model.predict_proba(features, task_type="METAL").shape == (len(targets), 3)
    path = tmp_path / "public.multiclass.snapshot"
    CatBoostMetalClassifier(**{**options, "iterations": 2}).fit(
        features, labels, save_snapshot=True, snapshot_file=path)
    resumed = CatBoostMetalClassifier(**options).fit(features, labels, save_snapshot=True, snapshot_file=path)
    complete = CatBoostMetalClassifier(**options).fit(features, labels)
    np.testing.assert_allclose(resumed.training_predictions_, complete.training_predictions_, atol=2e-6)


def test_64_class_validation_shares_bins_that_previously_exceeded_memory_limit(metal_device, tmp_path):
    classes, features, rows = 64, 512, 40001
    learn_bins = np.zeros((features, classes), np.uint8)
    validation_bins = np.zeros((features, rows), np.uint8)
    targets = np.arange(classes, dtype=np.float32)
    validation_targets = (np.arange(rows) % classes).astype(np.float32)
    empty = np.empty(0, np.uint32)
    bias = np.linspace(-0.3, 0.3, classes, dtype=np.float32)
    # The previous one-cursor-per-class layout rejected this otherwise modest
    # dataset solely because it duplicated the same feature matrix 64 times.
    assert validation_bins.nbytes * classes > 1 << 30
    path = tmp_path / "shared.matrix.snapshot"
    result = run_training(
        learn_bins, targets, empty, empty, classes=classes, objective="MultiClassOneVsAll",
        iterations=2, depth=0, learning_rate=0.2, l2_leaf_reg=3, score_function="L2",
        bias=bias, leaf_estimation_method="Gradient", eval_bins=validation_bins,
        eval_targets=validation_targets, use_best_model=False, save_snapshot=True, snapshot_file=path)
    stats = result.stats["validation"]
    assert stats["dataset_uploads"] == 1
    assert stats["bins_upload_bytes"] == validation_bins.nbytes
    assert stats["kernel_dispatches"] == 2
    assert stats["output_dimensions"] == classes
    assert stats["resident_bytes"] < 40 * 1024 * 1024
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["eval_predictions"]
        assert raw.shape == (rows, classes)
        expected = bias.copy()
        for leaves in result.leaf_values[:, 0]:
            expected = (expected + leaves).astype(np.float32)
        np.testing.assert_array_equal(raw, np.broadcast_to(expected, raw.shape))
        np.testing.assert_allclose(raw[0], result.predictions[0], atol=4e-8)
    assert result.evals_result["validation"]["MultiClassOneVsAll"][-1] == pytest.approx(
        metric(raw, validation_targets, objective="MultiClassOneVsAll"), abs=1e-7)


@pytest.mark.parametrize("bootstrap", ["No", "MVS"])
def test_public_categorical_default_four_permutation_snapshot_roundtrip(metal_device, tmp_path, bootstrap):
    from catboost_metal import CatBoostMetalRegressor

    rng = np.random.default_rng(772)
    categories = np.tile(np.arange(12), 21)
    rng.shuffle(categories)
    numeric = rng.normal(size=len(categories))
    features = np.column_stack(([f"group-{value}" for value in categories], numeric)).astype(object)
    targets = (3 * (categories % 2) + 0.2 * numeric).astype(np.float32)
    weights = rng.uniform(0.1, 2, len(targets)).astype(np.float32)
    settings = dict(depth=3, learning_rate=0.3, cat_features=[0], one_hot_max_size=2,
                    random_seed=67, random_strength=0, bootstrap_type=bootstrap,
                    **({"subsample": 0.7} if bootstrap == "MVS" else {}))
    fit = dict(sample_weight=weights, eval_set=(features, targets, weights), use_best_model=False)
    path = tmp_path / "categorical.four-permutation.snapshot"
    CatBoostMetalRegressor(iterations=2, **settings).fit(
        features, targets, save_snapshot=True, snapshot_file=path, **fit)
    resumed = CatBoostMetalRegressor(iterations=7, **settings).fit(
        features, targets, save_snapshot=True, snapshot_file=path, **fit)
    complete = CatBoostMetalRegressor(iterations=7, **settings).fit(features, targets, **fit)
    assert complete._layout.permutation_count == 4
    assert complete.training_stats_["permutation_count"] == 4
    assert complete.training_stats_["estimation_permutation"] == 3
    np.testing.assert_array_equal(resumed.training_predictions_, complete.training_predictions_)
    np.testing.assert_array_equal(resumed._result.leaf_values, complete._result.leaf_values)
    np.testing.assert_array_equal(resumed._result.split_features, complete._result.split_features)
    assert resumed.get_evals_result() == complete.get_evals_result()
    with np.load(path, allow_pickle=False) as snapshot:
        assert snapshot["permutation_predictions"].shape == (4, len(targets))
        np.testing.assert_array_equal(snapshot["permutation_predictions"][-1], complete.training_predictions_)
        assert snapshot["permutation_mvs_valid"].tolist() == ([1] * 4 if bootstrap == "MVS" else [0] * 4)
        used_features = snapshot["feature_penalty_used_features"]
        assert used_features.dtype == np.uint8 and used_features.shape == (len(complete._layout.borders),)
        assert used_features[list(complete._layout.ctrs)].any()
        raw_validation = snapshot["eval_predictions"]
    exported = resumed.predict(features, prediction_type="RawFormulaVal", task_type="METAL")
    np.testing.assert_allclose(raw_validation, exported, atol=5e-7, rtol=1e-6)
    assert resumed.get_evals_result()["validation"]["RMSE"][-1] == pytest.approx(
        metric(raw_validation, targets, weights), abs=1e-10)


@pytest.mark.parametrize("objective", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("bootstrap", ["No", "Bayesian", "Bernoulli", "Poisson"])
def test_multiclass_four_permutation_snapshot_continuation(metal_device, tmp_path, objective, bootstrap):
    from catboost_metal import _multiclass
    from catboost_metal._data import cuda_search_permutation

    rng = np.random.default_rng(2821)
    bins = rng.integers(0, 3, (2, 131), dtype=np.uint8)
    matrices = [bins]
    for _ in range(3):
        current = bins.copy()
        for feature in current:
            rng.shuffle(feature)
        matrices.append(current)
    targets = ((matrices[-1][0].astype(int) + matrices[-1][1]) % 3).astype(np.float32)
    weights = rng.uniform(0.1, 2, len(targets)).astype(np.float32)
    candidates = np.repeat(np.arange(2, dtype=np.uint32), 2)
    borders = np.tile(np.arange(2, dtype=np.uint32), 2)
    options = dict(iterations=7, depth=2, classes=3, learning_rate=0.3, l2_leaf_reg=2,
                   score_function="Cosine", objective=objective, sample_weight=weights,
                   bias=np.asarray([0.125, -0.25, 0.375], np.float32),
                   leaf_estimation_iterations=2, random_seed=772, iteration_offset=13,
                   bootstrap_type=bootstrap, subsample=0.6)
    with _multiclass.Session(bins, targets, candidates, borders, **options) as session:
        session.configure_permutations(matrices)
        for iteration in range(7):
            session.select_permutation(cuda_search_permutation(772, 13 + iteration, 4))
            session.step()
        expected, expected_state = session.result(), session.permutation_state
    path = tmp_path / "multiclass.permutations.snapshot"
    lifecycle = dict(permutation_bins=tuple(matrices), eval_bins=matrices[-1], eval_targets=targets,
                     eval_weight=weights, use_best_model=False, save_snapshot=True, snapshot_file=path)
    run_training(bins, targets, candidates, borders, **{**options, "iterations": 2}, **lifecycle)
    resumed = run_training(bins, targets, candidates, borders, **options, **lifecycle)
    full = run_training(bins, targets, candidates, borders, **options,
                        **{**lifecycle, "save_snapshot": False})
    for field in ("depths", "split_features", "split_bins", "split_types", "leaf_values",
                  "leaf_weights", "predictions", "rmse"):
        np.testing.assert_array_equal(getattr(resumed, field), getattr(expected, field), err_msg=field)
        np.testing.assert_array_equal(getattr(resumed, field), getattr(full, field), err_msg=field)
    assert resumed.evals_result == full.evals_result
    assert resumed.stats["validation"]["dataset_uploads"] == 2
    assert resumed.stats["validation"]["kernel_dispatches"] == 7
    with np.load(path, allow_pickle=False) as snapshot:
        assert snapshot["permutation_predictions"].shape == (4, len(targets), 3)
        np.testing.assert_array_equal(snapshot["permutation_predictions"], expected_state["predictions"])
        np.testing.assert_array_equal(snapshot["permutation_optimization_predictions"],
                                      expected_state["optimization_predictions"])
        np.testing.assert_array_equal(snapshot["permutation_mvs_lambdas"], np.zeros(4, np.float32))
        np.testing.assert_array_equal(snapshot["permutation_mvs_valid"], np.zeros(4, np.uint8))


@pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                    reason="Ordered training requires Apple Silicon")
@pytest.mark.parametrize("classification", [False, True])
def test_public_ordered_validation_snapshot_and_export(tmp_path, classification):
    from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRegressor

    values = np.tile(np.asarray([0, 1], np.float32), 48)
    features = np.column_stack((values, np.sin(np.arange(len(values)))))
    targets = values if classification else 2 * values - 1
    wrong = 1 - targets if classification else -targets
    estimator = CatBoostMetalClassifier if classification else CatBoostMetalRegressor
    options = dict(boosting_type="Ordered", permutation_count=4, depth=2, learning_rate=0.3,
                   l2_leaf_reg=1, random_strength=0, bootstrap_type="No", min_fold_size=8)
    stopped = estimator(iterations=10, **options).fit(
        features, targets, eval_set=(features, wrong), early_stopping_rounds=2, use_best_model=True)
    assert stopped.training_stats_["boosting_type"] == "Ordered"
    assert stopped.training_stats_["permutation_count"] == 4
    assert stopped.get_best_iteration() == 0 and stopped.tree_count_ == 1
    np.testing.assert_allclose(stopped.predict(features, prediction_type="RawFormulaVal", task_type="METAL"),
                               stopped.training_predictions_, atol=2e-6)
    path = tmp_path / "public.ordered.snapshot"
    fit_options = dict(eval_set=(features, targets), use_best_model=False)
    estimator(iterations=2, **options).fit(
        features, targets, save_snapshot=True, snapshot_file=path, **fit_options)
    resumed = estimator(iterations=6, **options).fit(
        features, targets, save_snapshot=True, snapshot_file=path, **fit_options)
    complete = estimator(iterations=6, **options).fit(features, targets, **fit_options)
    np.testing.assert_array_equal(resumed.training_predictions_, complete.training_predictions_)
    np.testing.assert_array_equal(resumed._result.leaf_values, complete._result.leaf_values)
    assert resumed.get_evals_result() == complete.get_evals_result()
    with np.load(path, allow_pickle=False) as snapshot:
        assert snapshot["ordered_cursors"].size > len(targets)
        assert snapshot["ordered_descriptors"].shape[1] == 4
        assert "permutation_predictions" not in snapshot
    exported = tmp_path / "ordered.json"
    resumed.save_model(exported, format="json")
    description = json.loads(exported.read_text())
    params = description["model_info"]["params"]
    assert params["boosting_options"]["boosting_type"] == "Ordered"
    assert params["boosting_options"]["data_partition"] == "FeatureParallel"
    assert params["boosting_options"]["permutation_count"] == 4
    np.testing.assert_allclose(resumed.predict(features, prediction_type="RawFormulaVal", task_type="METAL"),
                               resumed.predict(features, prediction_type="RawFormulaVal"), atol=2e-6)
