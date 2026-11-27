"""Standalone text inputs use native Metal training and the standard CBM reader."""
import os

import numpy as np
import pytest
from catboost import CatBoostError, CatBoostRegressor, Pool
from catboost_metal import CatBoostMetalClassifier, CatBoostMetalRanker, CatBoostMetalRegressor


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                                reason="requires the native Metal extension")


@pytest.mark.parametrize("declaration", ["constructor", "fit", "pool"])
def test_regression_uses_text_and_preserves_model(tmp_path, declaration):
    # Keep every non-text input constant so prediction differences must use text.
    rows = [[1., "type", "quiet bungalow" if i % 2 else "large apartment"] for i in range(96)]
    labels = np.asarray([5. if i % 2 else 1. for i in range(len(rows))])
    options = dict(iterations=3, depth=2, learning_rate=.5)
    validation = (rows[-16:], labels[-16:])
    if declaration == "pool":
        train = Pool(rows[:-16], labels[:-16], cat_features=[1], text_features=[2])
        validation = Pool(*validation, cat_features=[1], text_features=[2])
        model = CatBoostMetalRegressor(**options).fit(train, eval_set=validation)
    else:
        declaration_options = dict(text_features=[2])
        model = CatBoostMetalRegressor(**options, cat_features=[1],
                **(declaration_options if declaration == "constructor" else {}))
        model.fit(rows[:-16], labels[:-16], eval_set=validation,
                  **(declaration_options if declaration == "fit" else {}))
    assert model.training_stats_["backend"] == "METAL"
    assert model.training_stats_["kernel_dispatches"] > 0
    assert model.to_catboost().get_text_feature_indices() == [2]
    raw = model.predict(rows[-16:])
    assert raw[1] > raw[0]
    history = model.get_evals_result()["validation"]["RMSE"]
    assert np.sqrt(np.mean((raw - labels[-16:]) ** 2)) == pytest.approx(min(history), rel=1e-5)
    path = tmp_path / "text.cbm"
    model.save_model(path)
    np.testing.assert_array_equal(CatBoostRegressor().load_model(path).predict(rows[-16:]), raw)
    with pytest.raises(CatBoostError, match="(?i)text|numeric/categorical"):
        model.predict(rows[-16:], task_type="METAL")


def test_pool_text_schema_mismatches_fail_before_fit():
    rows = [["left", "right"]] * 16
    train = Pool(rows, list(range(16)), text_features=[0], cat_features=[1])
    model = CatBoostMetalRegressor(iterations=2)
    with pytest.raises(ValueError, match="text_features disagrees"):
        model.fit(train, text_features=[1])
    validation = Pool(rows, list(range(16)), cat_features=[0, 1])
    with pytest.raises(ValueError, match="disagrees"):
        model.fit(train, eval_set=validation)


@pytest.mark.parametrize("kind", [CatBoostMetalClassifier, CatBoostMetalRanker])
def test_named_text_and_calcer_options_through_shared_adapter(kind):
    labels = np.arange(96) % 2
    rows = [["hot coffee" if y else "cold tea"] for y in labels]
    groups = {"group_id": np.arange(96) // 8} if kind is CatBoostMetalRanker else {}
    train = Pool(rows, labels, text_features=[0], feature_names=["review"], **groups)
    calcers = ["BoW"] if kind is CatBoostMetalRanker else ["NaiveBayes"]
    model = kind(iterations=3, depth=2, text_features=["review"], feature_calcers=calcers)
    model.fit(train, text_features=[0])
    assert model.training_stats_["backend"] == "METAL"
    assert model.to_catboost().get_text_feature_indices() == [0]
    assert np.ptp(model.predict(train, prediction_type="RawFormulaVal")) > 0
