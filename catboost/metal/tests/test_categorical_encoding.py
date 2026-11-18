"""Check exact upstream hashing and standard-model categorical predictions."""

import json

import numpy as np
import pytest
from catboost import CatBoostRegressor, Pool

from catboost_metal._categorical import (
    UNSEEN_CATEGORY_BIN, cat_feature_hashes, categorical_feature_json,
    fit_one_hot, one_hot_split_json,
)


@pytest.fixture(autouse=True)
def prohibit_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Categorical encoding must not invoke a CPU trainer")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)


def test_hashes_match_upstream_constants_and_integer_strings():
    actual = cat_feature_hashes(["foo", "bar", "", "hello", "α😀", "long" * 40, 17, "17"])
    np.testing.assert_array_equal(actual[:6], [3741020925, 50123586, 797982799,
                                             4079208671, 1590307440, 3563060738])
    assert actual[6] == actual[7]
    assert actual.dtype == np.uint32


def test_training_and_unseen_encoding_is_stable():
    encoding, bins = fit_one_hot(["red", "blue", "green", "red"])
    other, other_bins = fit_one_hot(["green", "red", "blue"])
    assert encoding == other
    np.testing.assert_array_equal(encoding.transform(["green", "red", "blue"]), other_bins)
    assert bins[0] == bins[3]
    assert encoding.transform(["unseen"])[0] == UNSEEN_CATEGORY_BIN
    assert len(encoding.candidate_bins) == 3


def test_single_category_has_no_split_candidates():
    encoding, bins = fit_one_hot(["constant"] * 3)
    np.testing.assert_array_equal(bins, [0, 0, 0])
    assert len(encoding.candidate_bins) == 0
    assert "values" not in categorical_feature_json(0, 0, encoding)


def test_full_byte_range_reserves_unseen_bin():
    encoding, bins = fit_one_hot([f"category-{index}" for index in range(255)])
    assert set(bins) == set(range(255))
    assert encoding.transform(["novel"])[0] == 255
    with pytest.raises(ValueError, match="CTR training is required"):
        fit_one_hot([f"category-{index}" for index in range(256)])


@pytest.mark.parametrize("values", [[1.5], [np.nan], [None], [True], [["nested"]]])
def test_invalid_categories_fail_explicitly(values):
    with pytest.raises(ValueError):
        fit_one_hot(values)


@pytest.mark.parametrize("max_size", [0, 256, True, 2.5])
def test_invalid_one_hot_limit(max_size):
    with pytest.raises(ValueError, match="one_hot_max_size"):
        fit_one_hot(["a", "b"], max_size=max_size)


def test_standard_catboost_json_and_binary_models_use_original_categories(tmp_path):
    enc, _ = fit_one_hot(["foo", "bar", "α😀", "long" * 40])
    # One numeric split and then one categorical equality split. Numeric
    # borders precede all one-hot values in CatBoost's binary split array.
    hash_foo = int(cat_feature_hashes(["foo"])[0])
    category_bin = enc.hashes.index(hash_foo)
    model_json = {
        "features_info": {
            "float_features": [{"feature_index": 0, "flat_feature_index": 1,
                                "feature_id": "number", "has_nans": False,
                                "nan_value_treatment": "AsIs", "borders": [0.5]}],
            "categorical_features": [categorical_feature_json(0, 0, enc, "kind")],
        },
        "oblivious_trees": [{"splits": [
            {"split_type": "FloatFeature", "float_feature_index": 0, "border": 0.5, "split_index": 0},
            one_hot_split_json(0, category_bin, enc, 1 + category_bin)],
            "leaf_values": [10.0, 11.0, 12.0, 13.0], "leaf_weights": [1.0] * 4}],
        "scale_and_bias": [1.0, [0.0]],
    }
    json_path = tmp_path / "categorical.json"
    json_path.write_text(json.dumps(model_json))
    model = CatBoostRegressor().load_model(str(json_path), format="json")
    rows = [["foo", 0], ["foo", 1], ["bar", 0], ["novel", 1], ["α😀", 0], ["long" * 40, 1]]
    expected = [12.0, 13.0, 10.0, 11.0, 10.0, 11.0]
    np.testing.assert_array_equal(model.predict(rows), expected)
    assert model.get_cat_feature_indices() == [0]
    assert model.feature_names_ == ["kind", "number"]
    pool = Pool(rows, cat_features=[0], feature_names=["kind", "number"])
    np.testing.assert_array_equal(model.predict(pool), expected)
    for model_format in ("json", "cbm"):
        path = tmp_path / f"roundtrip.{model_format}"
        model.save_model(str(path), format=model_format)
        restored = CatBoostRegressor().load_model(str(path), format=model_format)
        np.testing.assert_array_equal(restored.predict(rows), expected)


def test_invalid_split_bin_is_rejected():
    encoding, _ = fit_one_hot(["a", "b"])
    with pytest.raises(ValueError, match="unknown category bin"):
        one_hot_split_json(0, 255, encoding, 0)
