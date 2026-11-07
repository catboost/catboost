"""Original feature layouts survive greedy JSON and CBM model roundtrips."""
import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
from catboost import CatBoost, CatBoostClassifier, CatBoostRegressor, Pool

from catboost_metal._categorical import OneHotEncoding, cat_feature_hashes, fit_one_hot
from catboost_metal._ctrs import CtrResult
from catboost_metal._data import CtrFeature, FeatureLayout, quantize_features
from catboost_metal._greedy_model import model_json


MISSING = np.iinfo(np.uint32).max


@pytest.fixture(autouse=True)
def forbid_cpu_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Model-layout checks must not fit a CPU model.")
    for cls in (CatBoost, CatBoostRegressor, CatBoostClassifier):
        monkeypatch.setattr(cls, "fit", forbidden)
    monkeypatch.setattr(CatBoost, "_fit", forbidden)


def tree(spec):
    """Build a flat non-symmetric tree from (feature, bin, type, left, right)."""
    nodes, values = [], []

    def append(value):
        index = len(nodes)
        nodes.append([0] * 6)
        if isinstance(value, tuple):
            feature, border, kind, left, right = value
            nodes[index] = [feature, border, kind, append(left), append(right), MISSING]
        else:
            nodes[index][-1] = len(values)
            values.append(float(value))
        return index

    append(spec)
    return SimpleNamespace(nodes=np.array(nodes, np.uint32), leaf_values=np.array(values),
                           leaf_weights=np.ones(len(values)))


def forest(*specs):
    return SimpleNamespace(trees=tuple(tree(spec) for spec in specs), stats={"device": "test metadata"})


def check_readers(tmp_path, document, raw, expected, *, cat_features=(), names=None):
    path = tmp_path / "greedy-layout.json"
    path.write_text(json.dumps(document, allow_nan=False))
    model = CatBoost().load_model(path, format="json")
    pool = Pool(raw, cat_features=list(cat_features), feature_names=names)
    np.testing.assert_allclose(model.predict(pool, prediction_type="RawFormulaVal"), expected, atol=1e-8, rtol=1e-8)
    assert model.get_cat_feature_indices() == list(cat_features)
    for format_ in ("cbm", "json"):
        path = tmp_path / ("roundtrip." + format_)
        model.save_model(path, format=format_)
        restored = CatBoost().load_model(path, format=format_)
        np.testing.assert_allclose(restored.predict(pool, prediction_type="RawFormulaVal"), expected, atol=1e-8, rtol=1e-8)
        if os.environ.get("CATBOOST_NATIVE_METAL_TESTS") == "1":
            np.testing.assert_allclose(restored.predict(pool, prediction_type="RawFormulaVal", task_type="GPU"),
                                       expected, atol=1e-8, rtol=1e-8)
        if names is not None:
            assert restored.feature_names_ == names
    return model


@pytest.mark.parametrize("nan_mode", ("Min", "Max"))
@pytest.mark.parametrize("training_has_nans", (False, True))
def test_numeric_nan_treatment_matches_training_layout(tmp_path, nan_mode, training_has_nans):
    learn = np.array([[-1], [0], [1], [np.nan if training_has_nans else 2]], np.float32)
    borders, _, _, _ = quantize_features(learn, 4, nan_mode)
    layout = FeatureLayout(borders, [training_has_nans], nan_mode, ["value"], {})
    result = forest(*[(0, index, 0, 0., float(1 << index)) for index in range(len(borders[0]))])
    document = model_json(result, borders, layout=layout, bias=.125)
    feature = document["features_info"]["float_features"][0]
    assert feature["has_nans"] == training_has_nans
    treatment = "AsTrue" if nan_mode == "Max" else "AsFalse"
    assert feature["nan_value_treatment"] == (treatment if training_has_nans else "AsIs")
    raw = np.array([[np.nan], [-2], [0], [.5], [3]], np.float32)
    expected = .125 + sum(np.where(np.isnan(raw[:, 0]), training_has_nans and nan_mode == "Max",
                                  raw[:, 0] > threshold) * (1 << index)
                          for index, threshold in enumerate(borders[0]))
    check_readers(tmp_path, document, raw, expected, names=["value"])


def test_one_hot_hashes_names_original_indices_and_unseen_routing(tmp_path):
    categories = ["α😀", "blue", "green"]
    encoding, _ = fit_one_hot(categories)
    borders = [np.empty(0, np.float32), np.array([.5], np.float32)]
    layout = FeatureLayout(borders, [False, False], "Forbidden", ["kind", "number"], {0: encoding})
    result = forest((1, 0, 0, (0, 0, 1, -1., 2.), (0, 2, 1, 3., 7.)))
    document = model_json(result, borders, layout=layout, feature_names=["category", "amount"])
    assert layout.names == ["kind", "number"]
    info = document["features_info"]
    assert info["float_features"][0]["feature_index"] == 0
    assert info["float_features"][0]["flat_feature_index"] == 1
    assert info["categorical_features"][0]["values"] == encoding.signed_hashes
    root = document["trees"][0]
    assert root["split"]["split_index"] == 0
    assert root["left"]["split"]["split_index"] == 1
    assert root["right"]["split"]["split_index"] == 3
    raw = [[category, number] for category in categories + ["unseen"] for number in (0, 1)]
    hashes = cat_feature_hashes([item[0] for item in raw])
    expected = [(-1 if hashed != encoding.hashes[0] else 2) if number == 0 else
                (3 if hashed != encoding.hashes[2] else 7) for (_, number), hashed in zip(raw, hashes)]
    check_readers(tmp_path, document, raw, expected, cat_features=[0], names=["category", "amount"])


def ctr_result(ctr_type, target_index=0, *, prior=.5):
    categories = ["red", "blue", "green"]
    hashes = cat_feature_hashes(categories)
    counts = np.array([4, 2, 3], np.uint32)
    sums = np.array([3, 0, 2] if ctr_type != "FloatTargetMeanValue" else [7, -1, 3], np.float32)
    order = np.argsort(hashes)
    result = CtrResult(hashes[order], sums[order], counts[order], np.empty(0, np.float32),
                      ctr_type, target_index, prior, 1., {})
    return categories, counts, sums, result


@pytest.mark.parametrize("ctr_type,target_index", [
    ("FeatureFreq", 0), ("Borders", 0), ("Buckets", 0), ("Buckets", 1), ("FloatTargetMeanValue", 0),
])
def test_ctr_descriptors_full_tables_and_target_indices(tmp_path, ctr_type, target_index):
    categories, counts, sums, stats = ctr_result(ctr_type, target_index)
    borders = [np.empty(0, np.float32), np.array([-.1, .25, .5, .75, 1.5], np.float32)]
    layout = FeatureLayout(borders, [False, False], "Forbidden", ["kind"],
        {0: OneHotEncoding(())}, {1: CtrFeature(0, stats, None, 0)})
    result = forest(*[(1, index, 0, 0., float(1 << index)) for index in range(len(borders[1]))])
    document = model_json(result, borders, layout=layout)
    descriptor = document["features_info"]["ctrs"][0]
    assert descriptor["ctr_type"] == ctr_type and descriptor["target_border_idx"] == target_index
    assert descriptor["identifier"] in document["ctr_data"]
    for index, exported in enumerate(document["trees"]):
        split = exported["split"]
        assert split["split_type"] == "OnlineCtr" and split["split_index"] == index
        assert split["ctr_target_border_idx"] == target_index
    numerator = np.r_[counts if ctr_type == "FeatureFreq" else sums, 0].astype(float)
    denominator = np.full(4, counts.sum(), dtype=float) if ctr_type == "FeatureFreq" else np.r_[counts, 0].astype(float)
    values = (numerator + .5) / (denominator + 1.)
    expected = sum((values > threshold) * (1 << index) for index, threshold in enumerate(borders[1]))
    check_readers(tmp_path, document, [[category] for category in categories + ["unseen"]],
                  expected, cat_features=[0], names=["kind"])


def test_mixed_numeric_one_hot_and_ctr_split_offsets(tmp_path):
    small, _ = fit_one_hot(["yes", "no"])
    categories, counts, _, stats = ctr_result("FeatureFreq", prior=0.)
    borders = [np.empty(0, np.float32), np.array([.5], np.float32),
               np.empty(0, np.float32), np.array([.3], np.float32)]
    layout = FeatureLayout(borders, [False] * 4, "Forbidden", ["small", "number", "large"],
        {0: small, 2: OneHotEncoding(())}, {3: CtrFeature(2, stats, None, 0)})
    result = forest((1, 0, 0, (0, 1, 1, 1., 2.), (3, 0, 0, 4., 8.)))
    document = model_json(result, borders, layout=layout)
    exported = document["trees"][0]
    assert exported["split"]["split_index"] == 0
    assert exported["left"]["split"]["split_index"] == 2
    assert exported["right"]["split"]["split_index"] == 3
    assert exported["right"]["split"]["split_type"] == "OnlineCtr"
    raw = [[s, n, large] for s in ("yes", "no", "unknown") for n in (0, 1)
           for large in categories + ["new"]]
    count_by_name = dict(zip(categories, counts))
    expected = []
    for s, number, large in raw:
        if number == 0:
            expected.append(2 if cat_feature_hashes([s])[0] == small.hashes[1] else 1)
        else:
            expected.append(8 if count_by_name.get(large, 0) / (counts.sum() + 1) > .3 else 4)
    check_readers(tmp_path, document, raw, expected, cat_features=[0, 2], names=layout.names)


def test_shared_ctr_identifier_can_reuse_full_table_across_priors(tmp_path):
    categories, counts, _, first = ctr_result("FeatureFreq", prior=0.)
    _, _, _, second = ctr_result("FeatureFreq", prior=1.)
    borders = [np.empty(0, np.float32), np.array([.35], np.float32), np.array([.35], np.float32)]
    layout = FeatureLayout(borders, [False] * 3, "Forbidden", ["kind"], {0: OneHotEncoding(())},
        {1: CtrFeature(0, first, None, 0), 2: CtrFeature(0, second, None, 0)})
    document = model_json(forest((1, 0, 0, 0., 1.), (2, 0, 0, 0., 2.)), borders, layout=layout)
    assert len(document["ctr_data"]) == 1
    assert len(document["features_info"]["ctrs"]) == 2
    values = np.r_[counts, 0]
    expected = (values / 10 > .35) + 2 * ((values + 1) / 10 > .35)
    check_readers(tmp_path, document, [[category] for category in categories + ["new"]], expected, cat_features=[0])
    conflicting = copy.deepcopy(layout)
    conflicting.ctrs[2].result.counts[0] += 1
    with pytest.raises(ValueError, match="same full statistics"):
        model_json(forest((1, 0, 0, 0., 1.)), borders, layout=conflicting)


@pytest.mark.parametrize("mode", ("Min", "Max"))
def test_root_only_model_keeps_original_feature_layout(tmp_path, mode):
    borders = [np.empty(0, np.float32), np.array([.5], np.float32)]
    layout = FeatureLayout(borders, [False, True], mode, ["category", "number"], {0: OneHotEncoding(())})
    document = model_json(forest(3.5), borders, layout=layout, bias=.5)
    check_readers(tmp_path, document, [["new", np.nan], ["another", 1.]], [4., 4.],
                  cat_features=[0], names=layout.names)


@pytest.mark.parametrize("change, message", [
    (lambda layout: setattr(layout, "has_nans", []), "match the supplied"),
    (lambda layout: setattr(layout, "nan_mode", "Unknown"), "nan_mode"),
    (lambda layout: layout.has_nans.__setitem__(1, True), "Forbidden"),
    (lambda layout: layout.categorical.__setitem__(2, OneHotEncoding(())), "categorical feature index"),
    (lambda layout: setattr(layout, "ctrs", {1: None}), "CTR feature index"),
    (lambda layout: setattr(layout, "names", ["only"]), "generated feature"),
])
def test_invalid_layout_is_rejected(change, message):
    borders = [np.empty(0, np.float32), np.array([.5], np.float32)]
    layout = FeatureLayout(borders, [False, False], "Forbidden", ["category", "number"], {0: OneHotEncoding(())})
    change(layout)
    with pytest.raises(ValueError, match=message):
        model_json(forest((1, 0, 0, 0., 1.)), borders, layout=layout)


def test_layout_split_types_and_unknown_categories_cannot_be_reinterpreted():
    encoding, _ = fit_one_hot(["yes", "no"])
    borders = [np.empty(0, np.float32), np.array([.5], np.float32)]
    layout = FeatureLayout(borders, [False, False], "Forbidden", ["category", "number"], {0: encoding})
    for split in ((0, 0, 0, 0., 1.), (1, 0, 1, 0., 1.), (0, 255, 1, 0., 1.)):
        with pytest.raises(ValueError, match="inconsistent"):
            model_json(forest(split), borders, layout=layout)
    with pytest.raises(ValueError, match="numeric"):
        model_json(forest((0, 0, 1, 0., 1.)), borders)
    with pytest.raises(ValueError, match="one string per original"):
        model_json(forest(1.), borders, layout=layout, feature_names=["wrong"])
