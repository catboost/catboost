"""Actual GPU validation and continuation for variable-node trees; no CPU fitting."""
import ctypes as ct
from dataclasses import replace
import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor

from catboost_metal import _greedy
from catboost_metal._greedy_inference import EvaluationCursor, EvaluationStats, predict_bins
from catboost_metal._greedy_training import run_training
from catboost_metal._training import _fingerprint, _json
from test_greedy_training import data, loss, route


@pytest.fixture(autouse=True)
def actual_metal_without_cpu_training(monkeypatch):
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU CatBoost fitting is forbidden in Metal tests")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)


def problem(objective="RMSE", policy="Lossguide"):
    bins, targets, weights, cf, cb = data(objective)
    args = (bins[:, :193], targets[:193], cf, cb)
    options = dict(iterations=8, depth=4, grow_policy=policy, max_leaves=9,
        learning_rate=.2, l2_leaf_reg=2.1, bias=.125, score_function="NewtonL2",
        objective=objective, sample_weight=weights[:193],
        eval_bins=bins[:, 193:], eval_targets=targets[193:], eval_weight=weights[193:],
        use_best_model=False, random_seed=817, objective_param=1)
    return args, options


@pytest.mark.parametrize("objective", ["RMSE", "Logloss"])
@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
def test_weighted_validation_resident_gpu_and_tree_ranges(objective, policy):
    args, options = problem(objective, policy)
    result = run_training(*args, **options)
    expected = np.full(options["eval_bins"].shape[1], options["bias"], np.float32)
    for index, tree in enumerate(result.trees):
        expected += tree.leaf_values[route(tree, options["eval_bins"])]
        expected_loss = loss(expected.astype(float), options["eval_targets"], options["eval_weight"], objective)
        assert result.evals_result["validation"][objective][index] == pytest.approx(expected_loss, rel=2e-6)
    np.testing.assert_array_equal(result.eval_predictions, expected)
    np.testing.assert_array_equal(predict_bins(args[0], result.trees, options["bias"]), result.predictions)
    np.testing.assert_array_equal(predict_bins(options["eval_bins"], result.trees, options["bias"]), expected)
    for start, end in ((0, 0), (0, 5), (2, 7), (8, 8)):
        reference = np.full(len(expected), options["bias"] if start == 0 else 0., np.float32)
        for tree in result.trees[start:end]:
            reference += tree.leaf_values[route(tree, options["eval_bins"])]
        np.testing.assert_array_equal(predict_bins(options["eval_bins"], result.trees,
            options["bias"], tree_start=start, tree_end=end), reference)
    assert result.stats["validation"]["dataset_uploads"] == 1
    assert result.stats["validation"]["bins_upload_bytes"] == options["eval_bins"].size
    assert result.stats["validation"]["kernel_dispatches"] >= len(result.trees)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
@pytest.mark.parametrize("objective", ["RMSE", "Logloss"])
def test_callback_snapshot_resume_is_exact_and_numeric(tmp_path, policy, objective):
    args, options = problem(objective, policy)
    complete = run_training(*args, **options)
    path = tmp_path / "greedy.npz"
    callbacks = []
    def callback(info):
        callbacks.append(info.iteration)
        assert len(info.metrics["learn"][objective]) == info.iteration
        info.metrics["learn"][objective].clear()  # Callback metrics are detached.
        return info.iteration < 3
    partial = run_training(*args, **options, save_snapshot=True, snapshot_file=path,
                           snapshot_interval=0, callback=callback)
    assert partial.stats["stop_reason"] == "callback" and callbacks == [1, 2, 3]
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[name].dtype.kind != "O" for name in archive.files)
        assert archive["nodes"].ndim == 2 and archive["nodes"].shape[1] == 6
        assert len(archive["node_offsets"]) == len(archive["leaf_offsets"]) == 4
    resumed = run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    assert resumed.resumed_iterations == 3 and resumed.trained_iterations == 8
    np.testing.assert_array_equal(resumed.predictions, complete.predictions)
    np.testing.assert_array_equal(resumed.eval_predictions, complete.eval_predictions)
    np.testing.assert_array_equal(resumed.loss, complete.loss)
    assert resumed.evals_result == complete.evals_result
    for left, right in zip(resumed.trees, complete.trees):
        np.testing.assert_array_equal(left.nodes, right.nodes)
        np.testing.assert_array_equal(left.leaf_values, right.leaf_values)
        np.testing.assert_array_equal(left.leaf_weights, right.leaf_weights)
    reread = run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    assert reread.resumed_iterations == 8
    np.testing.assert_array_equal(reread.predictions, complete.predictions)


@pytest.mark.parametrize("policy", ["Depthwise", "Lossguide", "Region"])
def test_early_stop_trim_keeps_full_snapshot_cursor(tmp_path, policy):
    args, options = problem(policy=policy)
    options.update(eval_bins=args[0], eval_targets=-args[1], eval_weight=options["sample_weight"],
                   bias=0., use_best_model=True, early_stopping_rounds=2)
    path = tmp_path / "trim.npz"
    result = run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    assert result.best_iteration == 0
    assert len(result.trees) == 1 and result.trained_iterations == 3
    assert result.stats["stop_reason"] == "early_stopping"
    np.testing.assert_array_equal(result.predictions, predict_bins(args[0], result.trees))
    assert len(result.evals_result["learn"]["RMSE"]) == 3
    with np.load(path, allow_pickle=False) as archive:
        assert len(archive["node_offsets"]) == 4
        saved_cursor = archive["predictions"].copy()
    assert not np.array_equal(saved_cursor, result.predictions)
    again = run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    assert again.resumed_iterations == again.trained_iterations == 3 and len(again.trees) == 1
    np.testing.assert_array_equal(again.predictions, result.predictions)


@pytest.mark.parametrize("field", ["random_seed", "objective_param", "sample_weight", "eval_weight", "bins"])
def test_snapshot_fingerprint_rejects_changed_inputs(tmp_path, field):
    args, options = problem()
    path = tmp_path / "fingerprint.npz"
    run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    if field == "bins":
        bins = args[0].copy(); bins[0, 0] ^= 1
        args = (bins,) + args[1:]
    elif field.endswith("weight"):
        options[field] = options[field].copy(); options[field][1] += .5
    elif field == "objective_param":
        options[field] = 2
    else:
        options[field] += 1
    with pytest.raises(ValueError, match="match|objective_param"):
        run_training(*args, **options, save_snapshot=True, snapshot_file=path)


@pytest.mark.parametrize("corruption", ["offsets", "graph", "predictions", "history", "best", "object", "checksum", "metadata"])
def test_snapshot_payload_is_validated_without_pickle(tmp_path, corruption):
    args, options = problem()
    path = tmp_path / "invalid.npz"
    run_training(*args, **options, save_snapshot=True, snapshot_file=path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        header = json.loads(archive["metadata"].item())
    header.pop("checksum")
    if corruption == "offsets": arrays["node_offsets"][1] = 0
    if corruption == "graph": arrays["nodes"][0, 3] = 0
    if corruption == "predictions": arrays["predictions"][0] = np.nan
    if corruption == "history": header["history"]["learn"]["RMSE"][0] += 1
    if corruption == "best": header["best_iteration"] = -1
    header["checksum"] = _fingerprint(arrays, header)
    if corruption == "object": arrays["nodes"] = arrays["nodes"].astype(object)
    if corruption == "checksum": arrays["predictions"][0] += 1
    np.savez(path, **arrays, metadata=np.asarray("[]" if corruption == "metadata" else _json(header)))
    with pytest.raises(ValueError):
        run_training(*args, **options, save_snapshot=True, snapshot_file=path)


def test_maximized_metric_selects_first_best():
    args, options = problem("Logloss")
    result = run_training(*args, **dict(options, use_best_model=True), eval_metric="Accuracy")
    scores = result.evals_result["validation"]["Accuracy"]
    assert result.best_iteration == int(np.argmax(scores))
    assert result.stats["metric_maximized"] is True
    assert len(result.trees) == result.best_iteration + 1
    assert result.best_score["validation"]["Accuracy"] == max(scores)


def test_external_training_cursor_survives_trim_and_resume(tmp_path):
    args, options = problem()
    initial = np.linspace(-.15, .2, len(args[1]), dtype=np.float32)
    options.update(initial_predictions=initial, eval_bins=args[0], eval_targets=-args[1],
        eval_weight=options["sample_weight"], bias=0., use_best_model=True,
        early_stopping_rounds=2, save_snapshot=True, snapshot_file=tmp_path / "baseline.npz")
    result = run_training(*args, **options)
    assert result.best_iteration == 0 and len(result.trees) == 1
    expected = initial + result.trees[0].leaf_values[route(result.trees[0], args[0])]
    np.testing.assert_array_equal(result.predictions, expected)
    resumed = run_training(*args, **options)
    np.testing.assert_array_equal(resumed.predictions, expected)


def test_training_metric_without_validation_and_zero_candidates_snapshot(tmp_path):
    bins = np.zeros((1, 17), np.uint8)
    targets = np.linspace(0, 1, 17, dtype=np.float32)
    options = dict(iterations=4, depth=3, learning_rate=.2, l2_leaf_reg=1.,
        bias=0., score_function="L2", eval_metric="MAE", save_snapshot=True,
        snapshot_file=tmp_path / "constant.npz")
    first = run_training(bins, targets, [], [], **options, callback=lambda info: info.iteration < 2)
    resumed = run_training(bins, targets, [], [], **options)
    assert first.trained_iterations == 2 and resumed.resumed_iterations == 2
    assert resumed.best_iteration == -1 and resumed.eval_predictions is None
    assert set(resumed.evals_result) == {"learn"}
    assert len(resumed.evals_result["learn"]["MAE"]) == 4
    assert all(len(tree.leaf_values) == 1 for tree in resumed.trees)


def test_resident_cursor_lifecycle_empty_rows_and_native_validation():
    args, options = problem()
    result = run_training(*args, **options)
    tree = result.trees[0]
    assert ct.sizeof(EvaluationStats) == 304
    with EvaluationCursor(args[0], bias=.25) as cursor:
        before = cursor.predictions()
        bad = tree.nodes.copy(); bad[0, 3] = 0
        with pytest.raises(ValueError): cursor.add_tree(replace(tree, nodes=bad))
        np.testing.assert_array_equal(cursor.predictions(), before)
        error = ct.create_string_buffer(2048)
        code = cursor._lib.cbm_greedy_evaluation_add_tree(cursor._handle,
            tree.nodes.ctypes.data_as(ct.POINTER(_greedy.Node)), len(tree.nodes),
            tree.leaf_values.ctypes.data_as(ct.POINTER(ct.c_float)), len(tree.leaf_values),
            before.ctypes.data_as(ct.POINTER(ct.c_float)), len(before) - 1, error, len(error))
        assert code != 0 and b"count" in error.value
        np.testing.assert_array_equal(cursor.predictions(), before)
        cursor.add_tree(tree)
        stats = cursor.stats
        assert stats["dataset_uploads"] == 1
        assert stats["tree_upload_bytes"] == tree.nodes.nbytes + tree.leaf_values.nbytes
    cursor.close()
    assert cursor.stats == stats
    with pytest.raises(RuntimeError, match="closed"): cursor.predictions()
    assert predict_bins(np.empty((4, 0), np.uint8), result.trees).shape == (0,)


@pytest.mark.parametrize("options", [{"early_stopping_rounds": 0}, {"snapshot_interval": -1},
    {"callback": 3}, {"save_snapshot": 1}, {"use_best_model": 1}, {"resume": 0},
    {"objective_param": .5}, {"random_seed": -1}, {"boosting_type": "Ordered"}])
def test_lifecycle_rejects_invalid_options(options):
    args, standard = problem()
    with pytest.raises((ValueError, TypeError)):
        run_training(*args, **dict(standard, **options))
