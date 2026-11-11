"""Exercise exported CTR tables with upstream inference, never CPU training."""

import json

import numpy as np
import pytest
from catboost import CatBoost, CatBoostRegressor, Pool

from catboost_metal._categorical import cat_feature_hashes, fit_one_hot, categorical_feature_json, one_hot_split_json
from catboost_metal._ctr_model import (
    combined_category_hashes, ctr_feature_json, ctr_identifier, ctr_table_json,
    online_ctr_split_json,
)


@pytest.fixture(autouse=True)
def prohibit_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CTR export must not invoke a CPU trainer")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def _model_json(descriptor, table):
    return {
        "features_info": {
            "categorical_features": [{"feature_index": 0, "flat_feature_index": 0, "feature_id": "kind"}],
            "ctrs": [descriptor],
        },
        "ctr_data": {descriptor["identifier"]: table},
        "oblivious_trees": [
            {"splits": [online_ctr_split_json(border, index, descriptor["target_border_idx"])],
             "leaf_values": [0.0, float(1 << index)], "leaf_weights": [1.0, 1.0]}
            for index, border in enumerate(descriptor["borders"])
        ],
        "scale_and_bias": [1.0, [0.0]],
    }


def _load_json(tmp_path, model_json):
    path = tmp_path / "ctr.json"
    path.write_text(json.dumps(model_json))
    return CatBoostRegressor().load_model(str(path), format="json")


def test_hash_sign_extension_and_exact_integer_arithmetic():
    raw = np.array([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], dtype=np.uint32)
    actual = combined_category_hashes(raw)
    multiplier = 0x4906BA494954CB65
    signed = [0, 1, 0x7FFFFFFF, -(1 << 31), -1]
    expected = [multiplier * multiplier * value % (1 << 64) for value in signed]
    np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.uint64))
    assert actual.dtype == np.uint64
    assert actual[-1] != multiplier * multiplier * 0xFFFFFFFF % (1 << 64)


@pytest.mark.parametrize("ctr_type,target_index", [
    ("FeatureFreq", 0), ("Borders", 0), ("Borders", 1),
    ("Buckets", 0), ("Buckets", 1), ("Buckets", 2), ("FloatTargetMeanValue", 0),
])
def test_original_categories_unseen_values_and_model_roundtrips(tmp_path, ctr_type, target_index):
    categories = ["foo", "bar", "hello"]
    hashes = cat_feature_hashes(categories)
    counts = np.array([4, 2, 3], dtype=np.uint32)
    class_counts = np.array([[1, 2, 1], [2, 0, 0], [0, 1, 2]], dtype=np.uint32)
    sums = np.array([7, -1, 3], dtype=np.float32)
    prior_num, prior_denom, shift, scale = 0.7, 1.3, 0.15, 2.0
    descriptor = ctr_feature_json(0, ctr_type, [-0.2, 0.1, 0.4, 0.7, 1.0, 1.3, 1.7, 2.2, 3.2],
                                  target_border_idx=target_index, prior_numerator=prior_num,
                                  prior_denominator=prior_denom, shift=shift, scale=scale)
    if ctr_type == "FeatureFreq":
        table = ctr_table_json(hashes, counts, ctr_type=ctr_type)
        numerator = np.r_[counts, 0].astype(np.float32)
        denominator = np.full(4, counts.sum(), dtype=np.float32)
    elif ctr_type == "FloatTargetMeanValue":
        table = ctr_table_json(hashes, counts, ctr_type=ctr_type, sums=sums)
        numerator, denominator = np.r_[sums, 0], np.r_[counts, 0].astype(np.float32)
    else:
        table = ctr_table_json(hashes, counts, ctr_type=ctr_type, class_counts=class_counts)
        good_counts = (class_counts[:, target_index] if ctr_type == "Buckets" else
                       class_counts[:, target_index + 1:].sum(axis=1))
        numerator = np.r_[good_counts, 0].astype(np.float32)
        denominator = np.r_[counts, 0].astype(np.float32)
    ctr_values = ((numerator + np.float32(prior_num)) / (denominator + np.float32(prior_denom)) +
                  np.float32(shift)) * np.float32(scale)
    expected = sum((ctr_values > border) * (1 << index)
                   for index, border in enumerate(descriptor["borders"]))
    model = _load_json(tmp_path, _model_json(descriptor, table))
    rows = [[value] for value in categories + ["never-seen-before"]]
    np.testing.assert_array_equal(model.predict(rows), expected)
    pool = Pool(rows, cat_features=[0], feature_names=["kind"])
    np.testing.assert_array_equal(model.predict(pool), expected)
    assert model.get_cat_feature_indices() == [0]
    for model_format in ("json", "cbm"):
        path = tmp_path / f"roundtrip.{model_format}"
        model.save_model(str(path), format=model_format)
        restored = CatBoostRegressor().load_model(str(path), format=model_format)
        np.testing.assert_array_equal(restored.predict(rows), expected)


@pytest.mark.parametrize("ctr_type,target_index", [("Borders", 0), ("Buckets", 1)])
def test_binary_positive_sums_match_class_histories(tmp_path, ctr_type, target_index):
    hashes = cat_feature_hashes(["foo", "bar"])
    counts = np.array([5, 3], dtype=np.uint32)
    positives = np.array([4, 1], dtype=np.float32)
    table = ctr_table_json(hashes, counts, ctr_type=ctr_type, sums=positives)
    expected = ctr_table_json(hashes, counts, ctr_type=ctr_type,
                              class_counts=np.array([[1, 4], [2, 1]], dtype=np.uint32))
    assert table == expected
    descriptor = ctr_feature_json(0, ctr_type, [0.4, 0.6], target_border_idx=target_index)
    model = _load_json(tmp_path, _model_json(descriptor, table))
    np.testing.assert_array_equal(model.predict([["foo"], ["bar"], ["new"]]), [3, 0, 1])


def test_float_and_one_hot_borders_precede_online_ctr_split_indices(tmp_path):
    encoding, _ = fit_one_hot(["foo", "bar", "hello"])
    descriptor = ctr_feature_json(0, "FeatureFreq", [0.3], prior_numerator=0)
    table = ctr_table_json(np.array(encoding.hashes, dtype=np.uint32),
                           np.array([1, 3, 6], dtype=np.uint32), ctr_type="FeatureFreq")
    first_hash = encoding.hashes[0]
    original = {int(hashed): name for hashed, name in zip(cat_feature_hashes(["foo", "bar", "hello"]),
                                                        ["foo", "bar", "hello"])}
    ctr_right_name = original[encoding.hashes[2]]
    one_hot_name = original[first_hash]
    model_json = {
        "features_info": {
            "float_features": [{"feature_index": 0, "flat_feature_index": 1, "feature_id": "number",
                                "has_nans": False, "nan_value_treatment": "AsIs", "borders": [0.5]}],
            "categorical_features": [categorical_feature_json(0, 0, encoding, "kind")],
            "ctrs": [descriptor],
        },
        "ctr_data": {descriptor["identifier"]: table},
        "oblivious_trees": [{"splits": [
            {"split_type": "FloatFeature", "float_feature_index": 0, "border": 0.5, "split_index": 0},
            one_hot_split_json(0, 0, encoding, 1),
            online_ctr_split_json(0.3, 1 + len(encoding.hashes)),
        ], "leaf_values": list(range(8)), "leaf_weights": [1.0] * 8}],
        "scale_and_bias": [1.0, [0.0]],
    }
    model = _load_json(tmp_path, model_json)
    rows = [[one_hot_name, 0], [one_hot_name, 1], [ctr_right_name, 0],
            [ctr_right_name, 1], ["new-category", 0]]
    np.testing.assert_array_equal(model.predict(rows), [2, 3, 4, 5, 0])


def test_identifier_is_shared_across_priors_and_target_borders():
    first = ctr_feature_json(3, "Buckets", [0.25], prior_numerator=0.5, target_border_idx=0)
    second = ctr_feature_json(3, "Buckets", [0.25], prior_numerator=1, target_border_idx=2)
    assert first["identifier"] == second["identifier"] == ctr_identifier(3, "Buckets")
    assert json.loads(first["identifier"]) == {
        "identifier": [{"cat_feature_index": 3, "combination_element": "cat_feature_value"}], "type": "Buckets"}


def test_fractional_target_means_are_rejected_before_corrupting_predictions():
    with pytest.raises(ValueError, match="Fractional FloatTargetMeanValue"):
        ctr_table_json(np.array([12], dtype=np.uint32), np.array([2], dtype=np.uint32),
                       ctr_type="FloatTargetMeanValue", sums=np.array([1.5], dtype=np.float32))


@pytest.mark.parametrize("hashes", [[-1], [2 ** 32], [1.0], [[1]], [True]])
def test_invalid_hashes(hashes):
    with pytest.raises(ValueError, match="hashes"):
        combined_category_hashes(hashes)


@pytest.mark.parametrize("overrides", [
    {"counts": np.array([-1, 2])}, {"counts": np.array([1.5, 2])},
    {"counts": np.array([1])}, {"hashes": np.array([12, 12], dtype=np.uint32)},
    {"counter_denominator": 1}, {"counter_denominator": 2 ** 31},
    {"sums": np.array([1, 2])},
])
def test_invalid_frequency_tables(overrides):
    args = {"hashes": np.array([12, 13], dtype=np.uint32),
            "counts": np.array([1, 2], dtype=np.uint32), "ctr_type": "FeatureFreq"}
    args.update(overrides)
    with pytest.raises(ValueError):
        ctr_table_json(**args)


@pytest.mark.parametrize("overrides", [
    {"prior_denominator": 0}, {"scale": 0}, {"target_border_idx": -1},
    {"borders": [0.5, 0.5]}, {"borders": [float("nan")]},
    {"borders": [0.5, 0.5 + 1e-10]}, {"prior_numerator": float("inf")},
])
def test_invalid_descriptor(overrides):
    args = {"cat_feature_index": 0, "ctr_type": "Borders", "borders": [0.5]}
    args.update(overrides)
    with pytest.raises(ValueError):
        ctr_feature_json(**args)


def test_class_counts_must_match_total_counts():
    with pytest.raises(ValueError, match="sum to counts"):
        ctr_table_json(np.array([12], dtype=np.uint32), np.array([2], dtype=np.uint32),
                       ctr_type="Borders", class_counts=np.array([[1, 3]], dtype=np.uint32))
