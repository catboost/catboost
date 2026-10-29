"""GPU prediction through the rebuilt native CatBoost extension, without fitting.

CATBOOST_NATIVE_METAL_TESTS=1 PYTHONPATH=/path/to/native/package python -m pytest ...
"""

import json
import os

import numpy as np
import pytest
from catboost import CatBoost, CatBoostError, CatBoostRegressor, Pool


pytestmark = pytest.mark.skipif(os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
                               reason="requires the rebuilt native Metal package")


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    monkeypatch.setattr(CatBoost, "_fit", lambda *a, **k: pytest.fail("Inference checks must not train models"))


def _numeric_model(depth=3, trees=9):
    rng = np.random.default_rng(519)
    return {"features_info": {"float_features": [
        {"feature_index": index, "flat_feature_index": index, "feature_id": f"x{index}",
         "borders": [.5], "has_nans": True, "nan_value_treatment": "AsFalse"} for index in range(depth)]},
        "oblivious_trees": [{"splits": [
            {"split_type": "FloatFeature", "float_feature_index": feature, "border": .5, "split_index": feature}
            for feature in range(index % (depth + 1))],
            "leaf_values": rng.normal(size=1 << (index % (depth + 1))).tolist()} for index in range(trees)],
        "scale_and_bias": [1.234567890123, [-.3141592653589]]}


def _category_model():
    foo, bar = -553946371, 50123586
    projection = [{"cat_feature_index": 0, "combination_element": "cat_feature_value"}]
    identifier = json.dumps({"identifier": projection, "type": "Borders"}, separators=(",", ":"))
    multiplier = 0x4906BA494954CB65
    table = {"hash_stride": 3, "counter_denominator": 0,
             "hash_map": [str(multiplier * multiplier * foo % 2**64), 1, 4,
                          str(multiplier * multiplier * bar % 2**64), 3, 1]}
    return {"features_info": {
        "float_features": [{"feature_index": 0, "flat_feature_index": 0, "feature_id": "number",
            "borders": [.5], "has_nans": True, "nan_value_treatment": "AsFalse"}],
        "categorical_features": [{"feature_index": 0, "flat_feature_index": 1,
            "feature_id": "category", "values": [foo, bar]}],
        "ctrs": [{"identifier": identifier, "elements": projection, "ctr_type": "Borders",
            "target_border_idx": 0, "prior_numerator": .5, "prior_denomerator": 1.,
            "shift": 0., "scale": 1., "borders": [.5]}]},
        "ctr_data": {identifier: table},
        "oblivious_trees": [{"splits": [
            {"split_type": "FloatFeature", "float_feature_index": 0, "border": .5, "split_index": 0},
            {"split_type": "OneHotFeature", "cat_feature_index": 0, "value": foo, "split_index": 1},
            {"split_type": "OnlineCtr", "border": .5, "ctr_target_border_idx": 0, "split_index": 3}],
            "leaf_values": (np.arange(8) / 10).tolist()}], "scale_and_bias": [1.125, [-.5]]}


def _load(tmp_path, data):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(data))
    return CatBoostRegressor().load_model(str(path), format="json")


@pytest.mark.parametrize("start,end", [(0, 0), (0, 5), (3, 7)])
@pytest.mark.parametrize("prediction_type", ["RawFormulaVal", "Probability", "Class", "Exponent"])
def test_native_gpu_numeric_model_ranges_and_prediction_types(tmp_path, start, end, prediction_type):
    data = _numeric_model()
    if prediction_type == "Class":
        data["model_info"] = {
            "params": json.dumps({"loss_function": {"type": "Logloss", "params": {}}}),
            "class_params": json.dumps({"class_label_type": "Integer", "class_names": [0, 1],
                                         "class_to_label": [0, 1], "classes_count": 0})}
    model = _load(tmp_path, data)
    rows = np.random.default_rng(701).normal(size=(1031, 3)).astype(np.float32)
    rows[:3] = [[.5, np.nan, .5], [np.inf, -np.inf, .5], [.5, .5, .5]]
    options = dict(ntree_start=start, ntree_end=end, prediction_type=prediction_type)
    expected = model.predict(rows, task_type="CPU", **options)
    actual = model.predict(rows, task_type="GPU", **options)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_native_gpu_categories_ctr_pool_and_cbm_roundtrip(tmp_path):
    model = _load(tmp_path, _category_model())
    rows = [[0., "foo"], [1., "foo"], [np.nan, "bar"], [1., "bar"], [.5, "unseen"]] * 207
    pool = Pool(rows, cat_features=[1], feature_names=["number", "category"])
    expected = model.predict(pool, task_type="CPU")
    np.testing.assert_allclose(model.predict(pool, task_type="GPU"), expected, rtol=0, atol=2e-15)
    path = tmp_path / "model.cbm"
    model.save_model(str(path))
    restored = CatBoostRegressor().load_model(str(path))
    np.testing.assert_allclose(restored.predict(rows, task_type="GPU"), expected, rtol=0, atol=2e-15)


def test_native_gpu_prequantized_pool_and_feature_names(tmp_path):
    model = _load(tmp_path, _numeric_model())
    rows = np.random.default_rng(442).integers(0, 2, size=(1031, 3)).astype(np.float32)
    pool = Pool(rows, feature_names=["x0", "x1", "x2"])
    pool.quantize(border_count=1)
    expected = model.predict(rows, task_type="CPU")
    np.testing.assert_allclose(model.predict(pool, task_type="GPU"), expected, rtol=2e-13, atol=2e-13)


def test_native_gpu_depth_twelve_and_constant_tree(tmp_path):
    data = _numeric_model(12, 13)
    model = _load(tmp_path, data)
    indexes = np.arange(4096, dtype=np.uint32)
    rows = ((indexes[:, None] >> np.arange(12, dtype=np.uint32)) & 1).astype(np.float32)
    expected = model.predict(rows, task_type="CPU")
    np.testing.assert_allclose(model.predict(rows, task_type="GPU"), expected, rtol=3e-12, atol=3e-12)


def _multidimensional_model(dimensions=3, *, depth=3, trees=9, objective="MultiClass"):
    data = _numeric_model(depth, trees)
    rng = np.random.default_rng(882)
    for tree in data["oblivious_trees"]:
        count = len(tree["leaf_values"])
        # Standard CatBoost stores adjacent class values within each leaf.
        tree["leaf_values"] = rng.normal(size=(count, dimensions)).reshape(-1).tolist()
    data["scale_and_bias"] = [-1.234567890123, np.linspace(-.7, .9, dimensions).tolist()]
    data["model_info"] = {
        "params": json.dumps({"loss_function": {"type": objective, "params": {}}}),
        "class_params": json.dumps({"class_label_type": "String",
            "class_names": [f"label-{index}" for index in range(dimensions)],
            "class_to_label": list(range(dimensions)), "classes_count": 0})}
    return data


@pytest.mark.parametrize("dimensions", [2, 3, 64])
@pytest.mark.parametrize("prediction_type", ["RawFormulaVal", "Probability", "Class", "LogProbability"])
@pytest.mark.parametrize("start,end", [(0, 0), (2, 7)])
def test_native_gpu_multidimensional_numeric_prediction(tmp_path, dimensions, prediction_type, start, end):
    model = _load(tmp_path, _multidimensional_model(dimensions))
    rows = np.random.default_rng(733).normal(size=(1031, 3)).astype(np.float32)
    rows[:3] = [[.5, np.nan, .5], [np.inf, -np.inf, .5], [.5, .5, .5]]
    options = dict(prediction_type=prediction_type, ntree_start=start, ntree_end=end)
    expected = model.predict(rows, task_type="CPU", **options)
    actual = model.predict(rows, task_type="GPU", **options)
    if prediction_type == "Class":
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual.shape == (len(rows), dimensions)
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-12)


@pytest.mark.parametrize("prediction_type", ["RawFormulaVal", "Probability", "Class", "LogProbability"])
def test_native_gpu_one_vs_all_multidimensional_prediction(tmp_path, prediction_type):
    model = _load(tmp_path, _multidimensional_model(3, objective="MultiClassOneVsAll"))
    rows = np.random.default_rng(661).normal(size=(1031, 3)).astype(np.float32)
    expected = model.predict(rows, task_type="CPU", prediction_type=prediction_type)
    actual = model.predict(rows, task_type="GPU", prediction_type=prediction_type)
    if prediction_type == "Class":
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=5e-12)
    if prediction_type == "Probability":
        # OVA is a separate sigmoid per class, not a normalized softmax.
        assert np.any(np.abs(expected.sum(axis=1) - 1.) > .01)


def test_native_gpu_multidimensional_ctr_model_and_roundtrip(tmp_path):
    data = _category_model()
    for tree in data["oblivious_trees"]:
        tree["leaf_values"] = [[value, -value, 2 * value + .1] for value in tree["leaf_values"]]
        tree["leaf_values"] = np.asarray(tree["leaf_values"]).reshape(-1).tolist()
    data["scale_and_bias"] = [1.125, [-.5, .75, -.25]]
    model = _load(tmp_path, data)
    rows = [[0., "foo"], [1., "foo"], [np.nan, "bar"], [1., "bar"], [.5, "unseen"]] * 207
    pool = Pool(rows, cat_features=[1], feature_names=["number", "category"])
    expected = model.predict(pool, task_type="CPU")
    np.testing.assert_allclose(model.predict(pool, task_type="GPU"), expected, rtol=3e-13, atol=3e-13)
    path = tmp_path / "multidimensional.cbm"
    model.save_model(str(path))
    restored = CatBoostRegressor().load_model(str(path))
    np.testing.assert_allclose(restored.predict(rows, task_type="GPU"), expected, rtol=3e-13, atol=3e-13)


def test_native_gpu_multidimensional_quantized_pool_and_multiple_blocks(tmp_path):
    model = _load(tmp_path, _multidimensional_model())
    rows = np.random.default_rng(975).integers(0, 2, size=(65539, 3)).astype(np.float32)
    pool = Pool(rows, feature_names=["x0", "x1", "x2"])
    pool.quantize(border_count=1)
    expected = model.predict(rows, task_type="CPU")
    np.testing.assert_allclose(model.predict(rows, task_type="GPU"), expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(model.predict(pool, task_type="GPU"), expected, rtol=3e-12, atol=3e-12)


def test_native_gpu_multidimensional_constant_and_deep_trees(tmp_path):
    for depth in [0, 12]:
        model = _load(tmp_path, _multidimensional_model(3, depth=depth, trees=depth + 1))
        indexes = np.arange(1 << depth, dtype=np.uint32)
        rows = ((indexes[:, None] >> np.arange(max(1, depth), dtype=np.uint32)) & 1).astype(np.float32)
        expected = model.predict(rows, task_type="CPU")
        np.testing.assert_allclose(model.predict(rows, task_type="GPU"), expected, rtol=4e-12, atol=4e-12)


def test_native_gpu_rejects_excessive_model_dimensions(tmp_path):
    data = _multidimensional_model(65, depth=1, trees=2)
    model = _load(tmp_path, data)
    with pytest.raises(CatBoostError, match=r"dimensions in \[1,64\]"):
        model.predict([[0.], [1.]], task_type="GPU")
