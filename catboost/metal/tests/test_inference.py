"""Actual Metal inference checked against fixtures and CatBoost's model reader.

Models are constructed directly; these tests do not train CPU CatBoost models.
"""

import ctypes as ct
import json
import platform

import numpy as np
import pytest
from catboost import CatBoostRegressor

from catboost_metal import _inference


@pytest.fixture(autouse=True)
def forbid_cpu_training(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Metal inference tests must not train CPU models")
    monkeypatch.setattr(CatBoostRegressor, "fit", forbidden)


@pytest.fixture
def metal():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")


def _arguments():
    return dict(bins=np.array([[0, 1, 2, 255]], np.uint8),
                depths=np.array([1], np.uint32), split_features=np.array([[0]], np.uint32),
                split_bins=np.array([[1]], np.uint32), leaf_values=np.array([[2.0, -3.0]]))


def _fixture_model(trees=73):
    rng = np.random.default_rng(1543)
    grids = [[-0.5, 0.0, 1.0], [-1.0, 0.25], [0.5]]
    offsets = np.cumsum([0] + [len(grid) for grid in grids])
    result = {"features_info": {"float_features": [
        {"feature_index": index, "flat_feature_index": index,
         "feature_id": str(index), "borders": grid, "has_nans": True,
         "nan_value_treatment": ["AsIs", "AsTrue", "AsFalse"][index]}
        for index, grid in enumerate(grids)]},
        "scale_and_bias": [1.2718281828459, [0.314159265358979]],
        "oblivious_trees": []}
    for index in range(trees):
        depth = index % 5
        splits = []
        for _ in range(depth):
            feature = int(rng.integers(0, 3))
            border = int(rng.integers(0, len(grids[feature])))
            splits.append({"split_type": "FloatFeature", "float_feature_index": feature,
                           "border": grids[feature][border],
                           "split_index": int(offsets[feature] + border)})
        result["oblivious_trees"].append({"splits": splits,
            "leaf_values": rng.normal(size=1 << depth).tolist(),
            "leaf_weights": np.ones(1 << depth).tolist()})
    return result


def test_bin_threshold_actual_depth_scale_and_batches(metal):
    arguments = _arguments()
    arguments.update(depths=np.array([1, 0, 1], np.uint32),
                     split_features=np.array([[0, 999], [999, 999], [0, 999]], np.uint32),
                     split_bins=np.array([[1, 0], [0, 0], [254, 0]], np.uint32),
                     leaf_values=np.array([[2.0, -3.0], [0.125, 999.0], [-7.0, 4.0]]))
    result, stats = _inference.predict_bins(**arguments, scale=2.5, bias=0.2,
                                            batch_size=3, return_stats=True)
    expected = (np.array([2, 2, -3, -3]) + .125 + np.array([-7, -7, -7, 4])) * 2.5 + .2
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-13)
    assert stats["device"] == "Apple M3 Pro" or stats["device"].startswith("Apple")
    assert stats["kernel_dispatches"] == 4
    assert stats["gpu_seconds"] >= 0


def test_onehot_equality_with_maximum_category(metal):
    arguments = _arguments()
    arguments.update(split_bins=np.array([[255]], np.uint32), split_types=np.ones((1, 1), np.uint8))
    result = _inference.predict_bins(**arguments)
    np.testing.assert_array_equal(result, [2, 2, 2, -3])


@pytest.mark.parametrize("start,end", [(0, None), (0, 17), (3, 64), (70, 73)])
def test_json_matches_standard_evaluator_including_nan_and_tree_range(metal, tmp_path, start, end):
    model = _fixture_model()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    rng = np.random.default_rng(112)
    data = rng.normal(size=(1031, 3)).astype(np.float32)
    data[:5] = [[np.nan, np.nan, np.nan], [-np.inf, np.inf, 0.5],
                [-0.5, -1.0, 0.5], [0.0, .25, .5], [1.0, 0.25, .5]]
    result = _inference.predict_model_json(model, data, tree_start=start, tree_end=end,
                                            batch_size=257)
    expected = standard.predict(data, ntree_start=start, ntree_end=0 if end is None else end)
    np.testing.assert_allclose(result, expected, rtol=2e-12, atol=2e-12)


def test_compensated_accumulation_preserves_cancelling_leaf_values(metal):
    # A plain float32 sum loses the unit term entirely; two components preserve
    # both cancellation inside a tile and the small residual between tiles.
    leaves = np.tile(np.array([1e10, 1.234567890123, -1e10]), 1001)
    result = _inference.predict_bins(np.empty((0, 7), np.uint8),
        np.zeros(len(leaves), np.uint32), np.empty((len(leaves), 0), np.uint32),
        np.empty((len(leaves), 0), np.uint32), leaves.reshape(-1, 1))
    expected = 1001 * 1.234567890123
    np.testing.assert_allclose(result, expected, rtol=3e-6, atol=0)
    assert abs(float(np.sum(leaves.astype(np.float32), dtype=np.float32)) - expected) > 100


def test_model_feature_mapping_and_extra_input_columns(metal, tmp_path):
    model = _fixture_model(11)
    path = tmp_path / "original.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    for feature in model["features_info"]["float_features"]:
        feature["flat_feature_index"] = 2 * feature["feature_index"]
    data = np.random.default_rng(331).normal(size=(127, 5)).astype(np.float32)
    np.testing.assert_allclose(_inference.predict_model_json(model, data), standard.predict(data[:, ::2]),
                               rtol=2e-12, atol=2e-12)


def test_empty_rows_and_tree_ranges_do_not_build(monkeypatch):
    def forbidden():
        raise AssertionError("Empty prediction should not compile or dispatch")
    monkeypatch.setattr(_inference, "build_library", forbidden)
    arguments = _arguments()
    arguments["bins"] = np.empty((1, 0), np.uint8)
    output, stats = _inference.predict_bins(**arguments, return_stats=True)
    assert output.shape == (0,) and output.dtype == np.float64
    assert stats["kernel_dispatches"] == 0
    arguments = _arguments()
    np.testing.assert_array_equal(_inference.predict_bins(**arguments, tree_end=0, bias=2.5), [2.5] * 4)
    np.testing.assert_array_equal(_inference.predict_bins(**arguments, tree_start=1, bias=2.5), [0.] * 4)


@pytest.mark.parametrize("change", [
    {"bins": [[0.0, 1.0]]}, {"bins": [[-1, 0]]}, {"bins": [[256, 0]]},
    {"depths": [2]}, {"depths": [17]}, {"depths": [True]},
    {"split_features": [[1]]}, {"split_features": [[-1]]}, {"split_bins": [[255]]},
    {"split_types": [[2]]}, {"split_types": [[0, 1]]},
    {"leaf_values": [[2.0]]}, {"leaf_values": [[np.inf, 0]]},
    {"leaf_values": [[1e39, 0]]}, {"leaf_values": [[1 + 2j, 0]]},
    {"bias": np.nan}, {"scale": np.inf}, {"scale": True},
    {"tree_start": -1}, {"tree_start": 2}, {"tree_start": True},
    {"tree_end": 2}, {"tree_start": 1, "tree_end": 0},
    {"batch_size": 0}, {"batch_size": 1.5}, {"return_stats": "yes"},
])
def test_validation_precedes_native_pointer_call(monkeypatch, change):
    def forbidden():
        raise AssertionError("Invalid input reached the compiler/native loader")
    monkeypatch.setattr(_inference, "build_library", forbidden)
    arguments = _arguments()
    arguments.update(change)
    with pytest.raises(ValueError):
        _inference.predict_bins(**arguments)


def test_native_rejects_short_input_before_pointer_read(metal):
    library = _inference._load(_inference.build_library())
    params = _inference.InferenceParams(2, 1, 1, 1, 2, 0, 1, 2, 1.0, 0.0)
    stats, error = _inference.InferenceStats(), ct.create_string_buffer(2048)
    # Null pointers paired with claimed counts must fail before any dereference.
    code = library.cbm_predict_bins(ct.byref(params), None, 1, None, 1,
        None, 1, None, 1, None, 1, None, 2, None, 2, ct.byref(stats), error, len(error))
    assert code == 1
    assert b"bins element count" in error.value


@pytest.mark.parametrize("mutate", [
    lambda model: model.pop("oblivious_trees"),
    lambda model: model["features_info"].update(categorical_features=[{}]),
    lambda model: model.update(scale_and_bias=[1, [0, 0]]),
    lambda model: model["oblivious_trees"][1].update(leaf_values=[1, 2, 3, 4]),
    lambda model: model["oblivious_trees"][1]["splits"][0].update(split_type="OnlineCtr"),
])
def test_json_rejects_unsupported_models_before_native(monkeypatch, mutate):
    model = _fixture_model(3)
    mutate(model)
    monkeypatch.setattr(_inference, "build_library", lambda: pytest.fail("Invalid JSON reached native"))
    with pytest.raises(ValueError):
        _inference.predict_model_json(model, np.ones((2, 3)))


def _categorical_fixture():
    from catboost_metal._categorical import cat_feature_hashes

    categories = ["foo", "bar", 17, "α😀"]
    hashes = [int(value) if value < 2**31 else int(value) - 2**32
              for value in cat_feature_hashes(categories)]
    model = {
        "features_info": {
            "float_features": [{"feature_index": 0, "flat_feature_index": 1,
                "feature_id": "number", "borders": [0.5], "has_nans": True,
                "nan_value_treatment": "AsFalse"}],
            "categorical_features": [
                {"feature_index": index, "flat_feature_index": index * 2,
                 "feature_id": f"category{index}", "values": hashes}
                for index in range(2)],
        },
        "oblivious_trees": [
            {"splits": [
                {"split_type": "FloatFeature", "float_feature_index": 0, "border": .5, "split_index": 0},
                {"split_type": "OneHotFeature", "cat_feature_index": 0, "value": hashes[0], "split_index": 1},
                {"split_type": "OneHotFeature", "cat_feature_index": 1, "value": hashes[2], "split_index": 7}],
             "leaf_values": (np.arange(8, dtype=np.float64) * .1234567890123).tolist()},
            {"splits": [{"split_type": "OneHotFeature", "cat_feature_index": 0,
                         "value": hashes[3], "split_index": 4}], "leaf_values": [-.75, .375]},
            {"splits": [], "leaf_values": [.234567890123456]},
        ],
        "scale_and_bias": [1.125, [-.314159265358979]],
    }
    return model


@pytest.mark.parametrize("start,end", [(0, None), (0, 1), (1, 3)])
def test_onehot_json_matches_standard_evaluator_original_values(metal, tmp_path, start, end):
    model = _categorical_fixture()
    path = tmp_path / "onehot.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    rows = [["foo", 0.0, 17], ["foo", .5, "17"], ["foo", 1.0, "bar"],
            ["novel", 1.0, "novel"], ["α😀", np.nan, "17"], [17, np.inf, "foo"],
            ["bar", -np.inf, "bar"], ["bar", 0.0, "bar"]]
    result, stats = _inference.predict_model_json(model, rows, tree_start=start,
        tree_end=end, batch_size=3, return_stats=True)
    expected = standard.predict(rows, ntree_start=start, ntree_end=0 if end is None else end)
    np.testing.assert_allclose(result, expected, rtol=2e-13, atol=2e-13)
    assert stats["kernel_dispatches"] == 6


def test_onehot_json_more_than_one_byte_of_known_categories(metal, tmp_path):
    from catboost_metal._categorical import cat_feature_hashes

    categories = [f"category-{index}" for index in range(300)]
    hashes = [int(value) if value < 2**31 else int(value) - 2**32
              for value in cat_feature_hashes(categories)]
    model = {"features_info": {"categorical_features": [
        {"feature_index": 0, "flat_feature_index": 0, "feature_id": "category", "values": hashes}]},
        "oblivious_trees": [{"splits": [
            {"split_type": "OneHotFeature", "cat_feature_index": 0,
             "value": hashes[index], "split_index": index} for index in (254, 255, 299)],
            "leaf_values": np.arange(8, dtype=np.float64).tolist()}],
        "scale_and_bias": [1.0, [0.0]]}
    rows = [[value] for value in categories + ["unknown"]]
    path = tmp_path / "many_onehot.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    result = _inference.predict_model_json(model, rows)
    expected = np.zeros(301)
    expected[[254, 255, 299]] = [1, 2, 4]
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result, standard.predict(rows))


@pytest.mark.parametrize("depth", [10, 12])
def test_json_deep_trees_beyond_training_depth_limit(metal, tmp_path, depth):
    leaf_ids = np.arange(1 << depth, dtype=np.uint32)
    rows = ((leaf_ids[:, None] >> np.arange(depth, dtype=np.uint32)) & 1).astype(np.float32)
    leaves = np.sin(np.arange(1 << depth)) * .125
    model = {
        "features_info": {"float_features": [
            {"feature_index": feature, "flat_feature_index": feature,
             "feature_id": str(feature), "has_nans": False,
             "nan_value_treatment": "AsIs", "borders": [.5]} for feature in range(depth)]},
        "oblivious_trees": [{"splits": [
            {"split_type": "FloatFeature", "float_feature_index": feature,
             "border": .5, "split_index": feature} for feature in range(depth)],
            "leaf_values": leaves.tolist()}],
        "scale_and_bias": [1.25, [.12345]],
    }
    path = tmp_path / "deep.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    result = _inference.predict_model_json(model, rows, batch_size=1031)
    np.testing.assert_allclose(result, leaves * 1.25 + .12345, rtol=0, atol=2e-15)
    np.testing.assert_allclose(result, standard.predict(rows), rtol=0, atol=2e-15)


def test_zero_depth_model_with_no_features_and_empty_rows(metal):
    model = {"features_info": {}, "oblivious_trees": [
        {"splits": [], "leaf_values": [.1234567890123]},
        {"splits": None, "leaf_values": [.3141592653589]}],
        "scale_and_bias": [-1.25, [.25]]}
    result = _inference.predict_model_json(model, np.empty((1031, 0)))
    np.testing.assert_allclose(result, (.1234567890123 + .3141592653589) * -1.25 + .25,
                               rtol=0, atol=2e-15)
    empty, stats = _inference.predict_model_json(model, np.empty((0, 0)), return_stats=True)
    assert empty.shape == (0,) and stats["kernel_dispatches"] == 0


@pytest.mark.parametrize("change", [
    lambda model: model["oblivious_trees"][0]["splits"][1].update(value=12345),
    lambda model: model["oblivious_trees"][0]["splits"][1].update(cat_feature_index=3),
    lambda model: model["features_info"]["categorical_features"][0].update(values=[1, 1]),
    lambda model: model["features_info"]["categorical_features"][0].update(values=[2**32]),
    lambda model: model["features_info"]["categorical_features"][0].update(flat_feature_index=1),
])
def test_onehot_json_rejects_invalid_metadata(monkeypatch, change):
    model = _categorical_fixture()
    change(model)
    monkeypatch.setattr(_inference, "build_library", lambda: pytest.fail("Invalid model reached GPU"))
    with pytest.raises(ValueError):
        _inference.predict_model_json(model, [["foo", 1.0, "bar"]])


@pytest.mark.parametrize("value", [1.5, np.nan, None, True])
def test_onehot_json_rejects_invalid_input_categories(value):
    with pytest.raises(ValueError, match="Categorical values"):
        _inference.predict_model_json(_categorical_fixture(), [[value, 1.0, "bar"]])


def _ctr_fixture(kind="Borders", target=0, priors=(.7,)):
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._ctr_model import ctr_feature_json, ctr_table_json, online_ctr_split_json

    hashes = cat_feature_hashes(["foo", "bar", "hello"])
    counts = np.array([4, 2, 3], np.uint32)
    kwargs = {}
    if kind == "FloatTargetMeanValue":
        kwargs["sums"] = np.array([7, -1, 3], np.float32)
    elif kind in ("Borders", "Buckets"):
        kwargs["class_counts"] = np.array([[1, 2, 1], [2, 0, 0], [0, 1, 2]], np.uint32)
    table = ctr_table_json(hashes, counts, ctr_type=kind, **kwargs)
    descriptors = [ctr_feature_json(0, kind, [-.2, .1, .4, .7, 1., 1.3, 1.7, 2.2, 3.2],
        prior_numerator=prior, prior_denominator=1.3, shift=.15, scale=2., target_border_idx=target)
        for prior in priors]
    model = {"features_info": {"categorical_features": [
        {"feature_index": 0, "flat_feature_index": 0, "feature_id": "category"}], "ctrs": descriptors},
        "ctr_data": {descriptors[0]["identifier"]: table},
        "oblivious_trees": [], "scale_and_bias": [1.125, [.125]]}
    offset = 0
    for descriptor in descriptors:
        for border in descriptor["borders"]:
            model["oblivious_trees"].append({"splits": [online_ctr_split_json(border, offset, target)],
                "leaf_values": [-.125, 2. ** (offset % 9)]})
            offset += 1
    return model


@pytest.mark.parametrize("kind,target", [("FeatureFreq", 0), ("Borders", 0), ("Borders", 1),
    ("Buckets", 0), ("Buckets", 1), ("Buckets", 2), ("FloatTargetMeanValue", 0)])
def test_all_cuda_ctr_json_types_match_standard_evaluator(metal, tmp_path, kind, target):
    model = _ctr_fixture(kind, target)
    path = tmp_path / "ctr.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    rows = [[name] for name in ["foo", "bar", "hello", "unknown", "foo"]]
    result = _inference.predict_model_json(model, rows, batch_size=3)
    np.testing.assert_array_equal(result, standard.predict(rows))


def test_ctr_multiple_priors_share_tables_but_keep_distinct_split_indices(metal, tmp_path):
    model = _ctr_fixture(priors=(.25, .75))
    path = tmp_path / "priors.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    rows = [[name] for name in ["foo", "bar", "hello", "unknown"]]
    np.testing.assert_array_equal(_inference.predict_model_json(model, rows), standard.predict(rows))


def test_ctr_combines_categories_numeric_and_onehot_projection(metal, tmp_path):
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._ctr_model import online_ctr_split_json

    rows = [["foo", 0., "x"], ["foo", 1., "x"], ["bar", 0., "y"], ["bar", 1., "y"]]
    first, second = cat_feature_hashes([row[0] for row in rows]), cat_feature_hashes([row[2] for row in rows])
    signed = lambda value: int(value) if value < 2**31 else int(value) - 2**32
    onehot_values = [signed(first[0]), signed(first[2])]
    elements = [{"combination_element": "cat_feature_value", "cat_feature_index": 0},
                {"combination_element": "cat_feature_value", "cat_feature_index": 1},
                {"combination_element": "float_feature", "float_feature_index": 0, "border": .5},
                {"combination_element": "cat_feature_exact_value", "cat_feature_index": 0, "value": onehot_values[0]}]
    identifier = json.dumps({"identifier": elements, "type": "Borders"}, separators=(",", ":"))
    descriptor = {"identifier": identifier, "elements": elements, "ctr_type": "Borders",
                  "prior_numerator": .5, "prior_denomerator": 1., "shift": 0., "scale": 1.,
                  "target_border_idx": 0, "borders": [.4, .6]}
    multiplier = 0x4906BA494954CB65
    history = []
    for index, row in enumerate(rows):
        key = 0
        for value in (signed(first[index]), signed(second[index]), int(row[1] > .5), int(row[0] == "foo")):
            key = (multiplier * (key + multiplier * value)) % 2**64
        history.extend([str(key), 4 - index, index])
    model = {"features_info": {
        "float_features": [{"feature_index": 0, "flat_feature_index": 1, "feature_id": "number",
                            "has_nans": False, "nan_value_treatment": "AsIs", "borders": [.5]}],
        "categorical_features": [{"feature_index": 0, "flat_feature_index": 0, "values": onehot_values},
                                 {"feature_index": 1, "flat_feature_index": 2}], "ctrs": [descriptor]},
        "ctr_data": {identifier: {"hash_stride": 3, "hash_map": history, "counter_denominator": 0}},
        "oblivious_trees": [{"splits": [online_ctr_split_json(.4, 3), online_ctr_split_json(.6, 4)],
                             "leaf_values": [0., 1., 2., 3.]}], "scale_and_bias": [1., [0.]]}
    path = tmp_path / "projection.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    rows.extend([["foo", 1., "y"], ["unknown", 1., "x"]])
    result = _inference.predict_model_json(model, rows)
    np.testing.assert_array_equal(result, [0., 0., 1., 3., 1., 1.])
    np.testing.assert_array_equal(result, standard.predict(rows))


def test_fractional_mean_ctr_json_preserves_float_history(metal):
    # The repository JSON importer has been corrected to GetDouble. Some older
    # installed wheels read these sums as integers, so use the explicit formula.
    model = _ctr_fixture("FloatTargetMeanValue")
    table = next(iter(model["ctr_data"].values()))
    for start, value in zip(range(0, len(table["hash_map"]), 3), [1.125, -.75, 2.25]):
        table["hash_map"][start + 1] = value
    desc = model["features_info"]["ctrs"][0]
    numerator = np.array([1.125, -.75, 2.25, 0.], np.float32)
    denominator = np.array([4., 2., 3., 0.], np.float32)
    ctr = ((numerator + np.float32(.7)) / (denominator + np.float32(1.3)) + np.float32(.15)) * np.float32(2.)
    expected = np.zeros(4)
    for index, border in enumerate(desc["borders"]):
        expected += np.where(ctr > np.float32(border), 2.**index, -.125)
    expected = expected * 1.125 + .125
    result = _inference.predict_model_json(model, [[v] for v in ["foo", "bar", "hello", "unknown"]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("change", [
    lambda model: model.pop("ctr_data"),
    lambda model: next(iter(model["ctr_data"].values())).update(hash_stride=2),
    lambda model: next(iter(model["ctr_data"].values()))["hash_map"].pop(),
    lambda model: next(iter(model["ctr_data"].values()))["hash_map"].__setitem__(1, -1),
    lambda model: model["features_info"]["ctrs"][0].update(prior_denomerator=0),
    lambda model: model["features_info"]["ctrs"][0].update(target_border_idx=3),
    lambda model: model["oblivious_trees"][0]["splits"][0].update(split_index=99),
    lambda model: model["oblivious_trees"][0]["splits"][0].update(border=.99),
])
def test_ctr_json_validation_precedes_native(monkeypatch, change):
    model = _ctr_fixture()
    change(model)
    monkeypatch.setattr(_inference, "build_library", lambda: pytest.fail("Malformed CTR model reached Metal"))
    with pytest.raises(ValueError):
        _inference.predict_model_json(model, [["foo"]])
