"""Acceptance checks for the rebuilt CatBoost native Metal Pool adapter.

Run with CATBOOST_NATIVE_METAL_TESTS=1 and the rebuilt extension on PYTHONPATH.
These deliberately invoke the standard CatBoost training API with task_type=GPU.
"""

import json
import os
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(
    not any(os.environ.get(name) == "1" for name in ("CATBOOST_NATIVE_METAL_TESTS", "CATBOOST_METAL_NATIVE_TESTS"))
    or platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Requires the rebuilt native Metal CatBoost extension",
)


@pytest.fixture(autouse=True)
def require_gpu_training(monkeypatch):
    original = CatBoost._fit

    def checked(self, *args, **kwargs):
        assert self._init_params.get("task_type") == "GPU", "Native tests may train only on Metal"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(CatBoost, "_fit", checked)


def _options(**extra):
    result = dict(task_type="GPU", iterations=15, depth=3, learning_rate=.25,
                  bootstrap_type="No", random_strength=0, leaf_estimation_backtracking="No",
                  verbose=False, allow_writing_files=False)
    result.update(extra)
    return result


def _roundtrip(model, pool, tmp_path):
    expected = model.predict(pool, prediction_type="RawFormulaVal")
    assert model.get_metadata()["metal_backend"] == "METAL"
    np.testing.assert_allclose(model.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"),
                               expected, atol=3e-5, rtol=3e-6)
    exported = None
    for model_format in ("json", "cbm"):
        path = tmp_path / f"native-cat.{model_format}"
        model.save_model(str(path), format=model_format)
        restored = model.__class__().load_model(str(path), format=model_format)
        np.testing.assert_allclose(restored.predict(pool, prediction_type="RawFormulaVal"), expected,
                                   atol=1e-6, rtol=1e-6)
        if model_format == "json":
            exported = json.loads(path.read_text())
    return exported


@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("one_hot_max_size,split_type", [(3, "OnlineCtr"), (4, "OneHotFeature")])
def test_native_onehot_threshold_pool_and_unseen_categories(tmp_path, prequantized, one_hot_max_size, split_type):
    categories = np.tile(np.arange(3), 40)
    X = [
        [["foo", "bar", "α😀"][value], float(value % 2)] for value in categories]
    y = np.array([-3, 2, 7], np.float32)[categories]
    train = Pool(X, y, cat_features=[0], feature_names=["kind", "number"])
    if prequantized:
        train.quantize()
    test = Pool([["foo", 0], ["bar", 1], ["α😀", 0], ["unseen", 1]],
                [-3, 2, 7, 0], cat_features=[0], feature_names=["kind", "number"])
    model = CatBoostRegressor(**_options(one_hot_max_size=one_hot_max_size)).fit(train, eval_set=test)
    assert model.get_cat_feature_indices() == [0]
    assert model.feature_names_ == ["kind", "number"]
    np.testing.assert_allclose(model.get_test_eval(), model.predict(test), atol=1e-6)
    exported = _roundtrip(model, test, tmp_path)
    # CUDA counts the unseen eval category in OnAll: three learn categories
    # plus one eval-only value require threshold four to retain one-hot mode.
    assert any(split["split_type"] == split_type for tree in exported["oblivious_trees"]
               for split in tree["splits"] or [])
    if prequantized:
        repeated = CatBoostRegressor(**_options(one_hot_max_size=one_hot_max_size)).fit(train, eval_set=test)
        np.testing.assert_array_equal(repeated.predict(test), model.predict(test))


@pytest.mark.parametrize("ctr_type", ["Borders", "Buckets", "FeatureFreq", "FloatTargetMeanValue"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_native_simple_ctrs_train_and_export_original_pool(tmp_path, ctr_type, permutation_count):
    rng = np.random.default_rng(619)
    category = np.repeat(np.arange(16), np.arange(9, 25))
    rng.shuffle(category)
    X = [[f"kind-{value}"] for value in category]
    y = category.astype(np.float32) + np.arange(len(category), dtype=np.float32) * .0137
    if ctr_type in ("Borders", "Buckets"):
        y = (category % 2).astype(np.float32) * 4 - 2
    train = Pool(X, y, cat_features=[0], feature_names=["kind"])
    heldout = Pool([["kind-0"], ["kind-1"], ["kind-10"], ["missing"]],
                   [0, 1, 10, 0], cat_features=[0], feature_names=["kind"])
    model = CatBoostRegressor(**_options(one_hot_max_size=2,
                                         simple_ctr=[f"{ctr_type}:Prior=0.5"],
                                         max_ctr_complexity=1)).set_params(permutation_count=permutation_count).fit(train, eval_set=heldout,
                                                                    use_best_model=False)
    assert model.get_cat_feature_indices() == [0]
    assert model.get_evals_result()["learn"]["RMSE"][-1] < model.get_evals_result()["learn"]["RMSE"][0]
    np.testing.assert_allclose(model.get_test_eval(), model.predict(heldout), atol=2e-6)
    exported = _roundtrip(model, heldout, tmp_path)
    assert any(split["split_type"] == "OnlineCtr" for tree in exported["oblivious_trees"]
               for split in tree["splits"] or [])
    assert {ctr["ctr_type"] for ctr in exported["features_info"]["ctrs"]} == {ctr_type}
    if ctr_type == "FloatTargetMeanValue":
        assert any(value != int(value) for table in exported["ctr_data"].values()
                   for value in table["hash_map"][1::3])


def test_native_default_ctrs_and_binary_pool(tmp_path):
    category = np.tile(np.arange(24), 20)
    y = (category % 2).astype(np.int32)
    train = Pool([[f"kind-{value}"] for value in category], y, cat_features=[0])
    test = Pool([["kind-0"], ["kind-1"], ["unknown"]], [0, 1, 0], cat_features=[0])
    model = CatBoostClassifier(**_options(one_hot_max_size=2)).fit(train, eval_set=test,
                                                                  use_best_model=False)
    assert model.get_evals_result()["learn"]["Logloss"][-1] < .5
    np.testing.assert_allclose(model.get_test_eval(), model.predict(test, prediction_type="RawFormulaVal"), atol=1e-6)
    _roundtrip(model, test, tmp_path)


def test_native_multiple_target_borders_and_per_feature_ctr(tmp_path):
    category = np.tile(np.arange(12), 24)
    X = [[f"kind-{value}", f"other-{value // 2}"] for value in category]
    y = (category % 3).astype(np.float32) * 3 + .01 * np.arange(len(category))
    train = Pool(X, y, cat_features=[0, 1])
    model = CatBoostRegressor(**_options(one_hot_max_size=2, ctr_target_border_count=2,
                                         simple_ctr=["Buckets:Prior=0.5"],
                                         per_feature_ctr=["1:FeatureFreq:Prior=0"],
                                         max_ctr_complexity=1)).fit(train)
    heldout = Pool([["kind-2", "other-1"], ["new", "unseen"]], cat_features=[0, 1])
    exported = _roundtrip(model, heldout, tmp_path)
    assert "Buckets" in {ctr["ctr_type"] for ctr in exported["features_info"]["ctrs"]}


@pytest.mark.parametrize("loss", ["MultiClass", "MultiClassOneVsAll"])
@pytest.mark.parametrize("ctr_type", ["Borders", "Buckets", "FeatureFreq", "FloatTargetMeanValue"])
@pytest.mark.parametrize("permutation_count", [1, 4])
def test_native_multiclass_categorical_pool_and_standard_roundtrip(tmp_path, loss, ctr_type, permutation_count):
    category = np.repeat(np.arange(12), np.tile([18, 24, 30], 4))
    np.random.default_rng(616).shuffle(category)
    labels = np.array(["alpha", "beta", "gamma"])[category % 3]
    weights = .5 + (np.arange(len(category)) % 7) / 5
    baseline = np.array([.5, -.2, .8])[None, :] + (np.arange(len(category)) % 4)[:, None] * .05
    train = Pool([[f"kind-{value}"] for value in category], labels, weight=weights,
                 baseline=baseline, cat_features=[0])
    heldout = Pool([["kind-0"], ["kind-1"], ["kind-2"], ["new"]],
                   ["alpha", "beta", "gamma", "alpha"], cat_features=[0],
                   baseline=np.array([.5, -.2, .8])[None, :] + np.arange(4)[:, None] * .05)
    model = CatBoostClassifier(**_options(loss_function=loss, one_hot_max_size=2,
                                          simple_ctr=[f"{ctr_type}:Prior=0.5"])).set_params(
        permutation_count=permutation_count).fit(train, eval_set=heldout, use_best_model=False)
    probabilities = model.predict_proba(heldout)
    assert probabilities.shape == (4, 3)
    assert np.isfinite(probabilities).all()
    assert ((probabilities > 0) & (probabilities < 1)).all()
    if loss == "MultiClass":
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-7)
    assert model.get_evals_result()["learn"][loss][-1] < model.get_evals_result()["learn"][loss][0]
    exported = _roundtrip(model, heldout, tmp_path)
    assert {ctr["ctr_type"] for ctr in exported["features_info"]["ctrs"]} == {ctr_type}


@pytest.mark.parametrize("permutation_count", [1, 4])
def test_native_prequantized_ctr_snapshot_restores_prefix_training_state(tmp_path, permutation_count):
    class StopAfter:
        def after_iteration(self, info):
            return info.iteration < 3

    category = np.tile(np.arange(24), 14)
    X = [[f"kind-{value}", float(row % 7)] for row, value in enumerate(category)]
    y = (category % 2).astype(np.float32) * 4 - 2 + np.arange(len(category), dtype=np.float32) * .001
    train = Pool(X, y, cat_features=[0])
    train.quantize(border_count=16)
    heldout = Pool([["kind-0", 0], ["kind-1", 1], ["new", 2]], [0, 1, 0], cat_features=[0])
    common = _options(iterations=10, random_seed=19, one_hot_max_size=2,
                      allow_writing_files=True, train_dir=str(tmp_path),
                      save_snapshot=True, snapshot_interval=0, snapshot_file="categorical.snapshot")
    partial = CatBoostRegressor(**common).set_params(permutation_count=permutation_count).fit(train, eval_set=heldout, use_best_model=False,
                                               callbacks=[StopAfter()])
    assert partial.tree_count_ == 3
    partial_path = tmp_path / "partial-ctr-model.json"
    partial.save_model(str(partial_path), format="json")
    assert any(split["split_type"] == "OnlineCtr"
               for tree in json.loads(partial_path.read_text())["oblivious_trees"]
               for split in tree["splits"] or [])
    resumed = CatBoostRegressor(**common).set_params(permutation_count=permutation_count).fit(train, eval_set=heldout, use_best_model=False)
    direct = CatBoostRegressor(**_options(iterations=10, random_seed=19, one_hot_max_size=2)).set_params(
        permutation_count=permutation_count).fit(
        train, eval_set=heldout, use_best_model=False)
    np.testing.assert_array_equal(resumed.get_tree_leaf_counts(), direct.get_tree_leaf_counts())
    np.testing.assert_allclose(resumed.get_leaf_values(), direct.get_leaf_values(), atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(resumed.get_evals_result()["learn"]["RMSE"],
                               direct.get_evals_result()["learn"]["RMSE"], atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(resumed.predict(heldout), direct.predict(heldout), atol=2e-6, rtol=2e-6)
    changed_rows = [list(row) for row in X]
    changed_rows[0][0] = "kind-1"
    changed = Pool(changed_rows, y, cat_features=[0])
    changed.quantize(border_count=16)
    with pytest.raises(Exception, match="snapshot.*data|data.*differ"):
        CatBoostRegressor(**common).set_params(permutation_count=permutation_count).fit(changed, eval_set=heldout, use_best_model=False)


@pytest.mark.parametrize("permutation_count", [1, 4])
def test_native_model_size_regularization_penalizes_unused_ctr_features(tmp_path, permutation_count):
    rng = np.random.default_rng(849)
    categories = np.tile(np.arange(12), 80)
    rng.shuffle(categories)
    targets = (categories % 2).astype(np.float32) * 2 - 1
    numeric = targets + rng.normal(0, 2.5, len(categories))
    train = Pool([[f"kind-{category}", value] for category, value in zip(categories, numeric)],
                 targets, cat_features=[0])
    split_types = []
    for regularization in (0, 20):
        model = CatBoostRegressor(**_options(iterations=1, depth=1, one_hot_max_size=2,
                                             simple_ctr=["Borders:Prior=0.5"],
                                             model_size_reg=regularization)).set_params(
            permutation_count=permutation_count).fit(train)
        path = tmp_path / f"model-size-{regularization}.json"
        model.save_model(str(path), format="json")
        tree = json.loads(path.read_text())["oblivious_trees"][0]
        assert len(tree["splits"]) == 1
        split_types.append(tree["splits"][0]["split_type"])
    assert split_types == ["OnlineCtr", "FloatFeature"]


@pytest.mark.parametrize("options,error", [
    ({"max_ctr_complexity": 2, "data_partition": "DocParallel"}, "FeatureParallel"),
    ({"counter_calc_method": "Full"}, "learn-only CTR"),
])
def test_native_unsupported_ctr_modes_reject_explicitly(options, error):
    train = Pool([["a"], ["b"], ["c"], ["a"], ["b"], ["c"]], [0, 1, 2, 0, 1, 2], cat_features=[0])
    with pytest.raises(Exception, match=error):
        CatBoostRegressor(**_options(one_hot_max_size=2, **options)).fit(train)
