"""CUDA fixed-prefix contracts exercised through native task_type='GPU'."""
import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, Pool

from test_native_greedy_api import StopAfter


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native Metal CatBoost package",
)


@pytest.fixture(autouse=True)
def require_gpu(monkeypatch):
    original = CatBoost._fit

    def guarded(self, *args, **kwargs):
        assert self.get_params().get("task_type") == "GPU"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", guarded)


def data(objective="RMSE", quantized=False):
    row = np.arange(160)
    x = np.empty((len(row), 4), object)
    x[:, 0] = row / 13  # ignored column still occupies a CUDA manager ID
    x[:, 1] = np.where((row // 8) % 2, "cat-a", "cat-b")
    x[:, 2] = (row % 2).astype(float)
    x[:, 3] = ((row // 2) % 2).astype(float)
    y = 20 * np.asarray(x[:, 3], float) + .01 * np.asarray(x[:, 2], float)
    if objective == "MultiClass":
        y = (row // 2) % 3
    pool = Pool(x, y, cat_features=[1], feature_names=["ignored", "category", "forced", "signal"])
    if quantized:
        pool.quantize(border_count=16, ignored_features=[0])
    return pool


def options(policy="Lossguide", **extra):
    result = dict(task_type="GPU", loss_function="RMSE", boosting_type="Plain",
                  grow_policy=policy, depth=3, iterations=3, learning_rate=.2,
                  random_seed=29, bootstrap_type="No", random_strength=0,
                  score_function="L2", leaf_estimation_iterations=1,
                  leaf_estimation_method="Gradient", leaf_estimation_backtracking="No",
                  boost_from_average=False, one_hot_max_size=2, border_count=16,
                  ignored_features=[0], fixed_binary_splits=[2, 2], permutation_count=1,
                  verbose=False, allow_writing_files=False)
    if policy == "Lossguide":
        result["max_leaves"] = 7
    result.update(extra)
    return result


def document(model, path):
    model.save_model(path, format="json")
    return json.loads(path.read_text())


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("objective", ("RMSE", "MultiClass"))
@pytest.mark.parametrize("quantized", (False, True))
def test_forced_binary_uses_manager_id_not_dense_or_float_index(tmp_path, policy, objective, quantized):
    pool = data(objective, quantized)
    model = CatBoost(options(policy, loss_function=objective)).fit(pool)
    assert model.get_metadata()["metal_backend"] == "METAL"
    saved = document(model, tmp_path / "forced.json")
    for tree in saved["trees"]:
        # Manager/flat ID 2 resolves to internal float ID 1, dense column 0.
        assert tree["split"]["split_type"] == "FloatFeature"
        assert tree["split"]["float_feature_index"] == 1
        assert tree["split"]["border"] == .5
        children = [tree["left"], tree["right"]]
        repeated = [child for child in children if "split" in child and child["split"]["float_feature_index"] == 1]
        assert repeated
        assert any(child.get("weight") == 0 for branch in repeated for child in (branch["left"], branch["right"]))
    prediction = model.predict(pool, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(model.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"),
                               prediction, rtol=4e-6, atol=4e-6)
    restored = CatBoost().load_model(tmp_path / "forced.json", format="json")
    np.testing.assert_allclose(restored.predict(pool, prediction_type="RawFormulaVal"), prediction, rtol=4e-6, atol=4e-6)


@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
@pytest.mark.parametrize("objective", ("RMSE", "MultiClass"))
@pytest.mark.parametrize("sampling", ("No", "Bernoulli"))
def test_fixed_splits_snapshot_resume_and_option_fingerprint(tmp_path, policy, objective, sampling):
    pool = data(objective)
    config = options(policy, loss_function=objective, iterations=6, bootstrap_type=sampling,
                     random_strength=.4, metric_period=2)
    if sampling == "Bernoulli":
        config["subsample"] = .8
    saved = dict(config, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_interval=0, snapshot_file="fixed.snapshot")
    partial = CatBoost(saved).fit(pool, eval_set=pool, callbacks=[StopAfter(2)], use_best_model=False)
    assert partial.tree_count_ == 2
    resumed = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False)
    direct = CatBoost(config).fit(pool, eval_set=pool, use_best_model=False)
    for method in ("get_tree_leaf_counts", "get_leaf_values", "get_leaf_weights", "get_test_eval"):
        np.testing.assert_array_equal(getattr(resumed, method)(), getattr(direct, method)())
    assert resumed.get_evals_result() == direct.get_evals_result()
    with pytest.raises(CatBoostError, match="(?i)snapshot.*differ|differ.*snapshot|parameters"):
        CatBoost(saved | dict(fixed_binary_splits=[3, 3])).fit(pool, eval_set=pool, use_best_model=False)


@pytest.mark.parametrize("requested", ([0], [1], [99]))
def test_fixed_splits_reject_ignored_categorical_and_unknown_ids(requested):
    with pytest.raises(CatBoostError, match="(?i)fixed.*float|fixed split feature"):
        CatBoost(options(fixed_binary_splits=requested)).fit(data())


def test_fixed_splits_reject_nonbinary_feature():
    pool = Pool(np.column_stack([np.arange(40), np.arange(40) % 2]), np.arange(40, dtype=float))
    with pytest.raises(CatBoostError, match="(?i)fixed.*binary|borders"):
        CatBoost(options(ignored_features=[], fixed_binary_splits=[0])).fit(pool)


@pytest.mark.parametrize("partition", ("DocParallel", "FeatureParallel"))
def test_scalar_symmetric_cuda_ignores_fixed_option(partition):
    pool = data()
    config = options("SymmetricTree", data_partition=partition, fixed_binary_splits=[999])
    ignored = CatBoost(config).fit(pool)
    empty = CatBoost(config | dict(fixed_binary_splits=[])).fit(pool)
    np.testing.assert_array_equal(ignored.get_leaf_values(), empty.get_leaf_values())
    np.testing.assert_array_equal(ignored.predict(pool), empty.predict(pool))


def test_vector_symmetric_cuda_rejects_fixed_option():
    with pytest.raises(CatBoostError, match="(?i)fixed.*symmetric"):
        CatBoost(options("SymmetricTree", loss_function="MultiClass")).fit(data("MultiClass"))


@pytest.mark.parametrize("objective", ("RMSE", "MultiClass"))
def test_region_rejects_cuda_incompatible_branching_prefix(objective):
    with pytest.raises(CatBoostError, match="(?i)region.*branching prefix"):
        CatBoost(options("Region", loss_function=objective, depth=4, fixed_binary_splits=[2, 3, 2])).fit(data(objective))


@pytest.mark.parametrize("objective", ("PairLogit", "YetiRank"))
@pytest.mark.parametrize("policy", ("Depthwise", "Lossguide", "Region"))
def test_query_fixed_root_and_completed_snapshot(tmp_path, objective, policy):
    row = np.arange(96)
    x = np.column_stack([row % 2, (row // 2) % 2]).astype(float)
    pool = Pool(x, ((row // 2) % 2).astype(float), group_id=row // 8)
    config = options(policy, loss_function=objective, ignored_features=[], fixed_binary_splits=[0],
                     leaf_estimation_method="Newton", iterations=3)
    saved = dict(config, allow_writing_files=True, train_dir=str(tmp_path), save_snapshot=True,
                 snapshot_interval=0, snapshot_file="query-fixed.snapshot")
    model = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False)
    resumed = CatBoost(saved).fit(pool, eval_set=pool, use_best_model=False)
    np.testing.assert_array_equal(model.get_leaf_values(), resumed.get_leaf_values())
    for tree in document(model, tmp_path / "query.json")["trees"]:
        assert tree["split"]["float_feature_index"] == 0
