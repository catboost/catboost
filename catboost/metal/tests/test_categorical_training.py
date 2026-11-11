"""Train original categorical inputs on Metal and verify standard-model export."""

import json
import platform

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor

from catboost_metal import CatBoostMetalRegressor, CatBoostMetalClassifier


pytestmark = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                                reason="Categorical training requires Apple Silicon")


@pytest.fixture(autouse=True)
def prohibit_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Categorical training must execute on Metal")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def test_one_hot_training_keeps_original_categories_and_unseen_routing(tmp_path):
    names = np.array(["foo", "bar", "α😀"], dtype=object)
    codes = np.tile(np.arange(3), 31)
    X, y = names[codes, None], np.array([-3, 2, 7], np.float32)[codes]
    model = CatBoostMetalRegressor(iterations=12, depth=3, learning_rate=.3, l2_leaf_reg=.1,
                                   cat_features=[0]).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0] * .1
    np.testing.assert_allclose(model.predict(X), model.training_predictions_, atol=3e-5)
    holdout = [["foo"], ["bar"], ["α😀"], ["never-before"]]
    expected = model.predict(holdout)
    np.testing.assert_allclose(model.predict(holdout, task_type="METAL"), expected, atol=3e-5)
    path = tmp_path / "onehot.json"
    model.save_model(path, format="json")
    saved = json.loads(path.read_text())
    assert any(split["split_type"] == "OneHotFeature" for tree in saved["oblivious_trees"]
               for split in tree["splits"] or [])
    assert model.to_catboost().get_cat_feature_indices() == [0]


@pytest.mark.parametrize("classification", [False, True])
def test_borders_ctr_training_and_cpu_gpu_saved_model_predictions(tmp_path, classification):
    rng = np.random.default_rng(402)
    category = np.tile(np.arange(32), 20)
    rng.shuffle(category)
    X = np.column_stack(([f"kind-{value}" for value in category], rng.normal(size=len(category)))).astype(object)
    y = (category % 2).astype(np.float32)
    if not classification:
        y = 4 * y - 2 + .05 * X[:, 1].astype(np.float32)
    estimator = CatBoostMetalClassifier if classification else CatBoostMetalRegressor
    model = estimator(iterations=20, depth=3, learning_rate=.25, l2_leaf_reg=.2,
                      cat_features=[0], one_hot_max_size=2, ctr_type="Borders", random_seed=7).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0] * .65
    assert len(model._layout.ctrs) == 1
    assert model.n_features_in_ == 2
    assert len(model._layout.borders) == 3
    ctr = next(iter(model._layout.ctrs.values())).result
    assert ctr.stats["kernel_dispatches"] > 0
    # Training uses exclusive prefix values; prediction uses the full table.
    from catboost_metal._categorical import cat_feature_hashes
    assert np.any(ctr.values != ctr.full_values(cat_feature_hashes(X[:, 0])))
    heldout = [["kind-0", -1], ["kind-1", 1], ["kind-30", 0], ["kind-31", 0], ["unseen-kind", 0]]
    expected = model.predict(heldout, prediction_type="RawFormulaVal")
    actual = model.predict(heldout, prediction_type="RawFormulaVal", task_type="METAL")
    np.testing.assert_allclose(actual, expected, atol=5e-5)
    for model_format in ("json", "cbm"):
        path = tmp_path / f"categorical.{model_format}"
        model.save_model(path, format=model_format)
        restored = model.to_catboost().__class__().load_model(str(path), format=model_format)
        np.testing.assert_allclose(restored.predict(heldout, prediction_type="RawFormulaVal"), expected, atol=1e-7)
        if model_format == "json":
            saved = json.loads(path.read_text())
            assert saved["features_info"]["float_features"][0]["flat_feature_index"] == 1
            assert len(saved["features_info"]["float_features"]) == 1
            assert any(split["split_type"] == "OnlineCtr" for tree in saved["oblivious_trees"]
                       for split in tree["splits"] or [])


def test_feature_frequency_ctr_and_all_categorical_inputs(tmp_path):
    category = np.repeat(np.arange(16), np.arange(1, 17))
    X = np.asarray([[f"kind-{value}"] for value in category], dtype=object)
    y = category.astype(np.float32)
    model = CatBoostMetalRegressor(iterations=15, depth=3, learning_rate=.3, cat_features=[0],
                                   one_hot_max_size=2, ctr_type="FeatureFreq", ctr_prior=0).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0] * .3
    np.testing.assert_allclose(model.predict(X), model.training_predictions_, atol=4e-5)
    heldout = [["kind-0"], ["kind-10"], ["kind-15"], ["novel"]]
    np.testing.assert_allclose(model.predict(heldout, task_type="METAL"), model.predict(heldout), atol=4e-5)
    path = tmp_path / "frequency.json"
    model.save_model(path, format="json")
    restored = CatBoostRegressor().load_model(str(path), format="json")
    assert restored.get_cat_feature_indices() == [0]


def test_mixed_one_hot_and_ctr_offsets_across_original_features():
    rows = 384
    category = np.tile(np.arange(16), rows // 16)
    small = np.where(np.arange(rows) % 3 == 0, "yes", "no")
    numeric = np.sin(np.arange(rows) * .27)
    X = np.column_stack((small, numeric, [f"cat-{value}" for value in category])).astype(object)
    y = ((category % 2) * 5 + (small == "yes") * 3 + numeric).astype(np.float32)
    model = CatBoostMetalRegressor(iterations=20, depth=4, learning_rate=.25, cat_features=[0, 2],
                                   one_hot_max_size=2, random_seed=3).fit(X, y)
    assert model.n_features_in_ == 3
    assert model.to_catboost().get_cat_feature_indices() == [0, 2]
    heldout = [["yes", .1, "cat-1"], ["no", -.2, "cat-4"], ["unseen", .7, "missing"]]
    np.testing.assert_allclose(model.predict(heldout, task_type="METAL"), model.predict(heldout), atol=4e-5)
