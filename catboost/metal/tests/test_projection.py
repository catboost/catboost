"""Real GPU projection hashes, full-width grouping, and compound model lookup."""

import ctypes as ct
import json
import platform

import numpy as np
import pytest

from catboost_metal._projection import group_projection, _build_library, _load, _Params, _Stats


METAL = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                          reason="Projection GPU tests require Apple Silicon")
MULTIPLIER = 0x4906BA494954CB65
MASK = (1 << 64) - 1


def _reference(cats, bins, types, features, thresholds):
    hashes = []
    for row in range(cats.shape[1]):
        result = 0
        for kind, feature, threshold in zip(types, features, thresholds):
            if kind == 0:
                value = int(cats[feature, row])
                if value >= 1 << 31:
                    value -= 1 << 32
            else:
                value = int(bins[feature, row] > threshold if kind == 1 else bins[feature, row] == threshold)
            result = (MULTIPLIER * (result + MULTIPLIER * value)) & MASK
        hashes.append(result)
    return np.asarray(hashes, np.uint64)


@METAL
@pytest.mark.parametrize("rows", [0, 1, 31, 257, 1031, 65539])
def test_compound_hash_and_stable_grouping_match_integer_reference(rows):
    rng = np.random.default_rng(104)
    cats = rng.integers(0, 1 << 32, (3, rows), dtype=np.uint32)
    if rows > 3:
        cats[:, 1::3] = cats[:, :1]
    bins = rng.integers(0, 5, (2, rows), dtype=np.uint8)
    types, features, thresholds = [0, 0, 1, 2], [0, 2, 1, 0], [0, 0, 2, 3]
    permutation = rng.permutation(rows)
    expected = _reference(cats, bins, types, features, thresholds)
    order = np.asarray(sorted(permutation, key=lambda row: int(expected[row])), np.uint32)
    result = group_projection(cats, bins, component_types=types, component_features=features,
                              component_thresholds=thresholds, permutation=permutation)
    np.testing.assert_array_equal(result.row_hashes, expected)
    np.testing.assert_array_equal(result.sorted_hashes, expected[order])
    np.testing.assert_array_equal(result.sorted_rows, order)
    np.testing.assert_array_equal(result.unique_hashes[result.row_bins], expected)
    np.testing.assert_array_equal(result.row_bins[result.sorted_rows], result.category_bins)
    assert result.stats["backend"] == "Metal"
    assert result.stats["hash_bits"] == 64
    assert result.stats["radix_passes"] == (16 if rows else 0)


@METAL
def test_low32_collisions_stay_distinct_and_true_duplicates_keep_history(monkeypatch):
    # M^2 * (M * a + b): these distinct tuples have the same low 32 bits.
    # A truncated grouping key would collapse their different model hashes.
    cats = np.array([[0, 1, 0, 1], [0, (-MULTIPLIER) & 0xffffffff, 0, (-MULTIPLIER) & 0xffffffff]], np.uint32)
    permutation = np.array([3, 2, 1, 0], np.uint32)
    expected = _reference(cats, np.empty((0, 4), np.uint8), [0, 0], [0, 1], [0, 0])
    assert expected[0] != expected[1]
    assert int(expected[0]) & 0xffffffff == int(expected[1]) & 0xffffffff

    def forbidden(*args, **kwargs):
        raise AssertionError("Projection grouping must not use CPU sorting")

    with monkeypatch.context() as patch:
        for name in ("sort", "argsort", "lexsort", "unique"):
            patch.setattr(np, name, forbidden)
        result = group_projection(cats, permutation=permutation)
    np.testing.assert_array_equal(result.row_hashes, expected)
    assert len(result.unique_hashes) == 2
    for value in result.unique_hashes:
        actual = result.sorted_rows[result.sorted_hashes == value]
        np.testing.assert_array_equal(actual, permutation[expected[permutation] == value])


@METAL
def test_signed_category_extension_and_predicate_only_projection():
    cats = np.array([[0, 0x7fffffff, 0x80000000, 0xffffffff]], np.uint32)
    result = group_projection(cats)
    expected = [(MULTIPLIER * MULTIPLIER * value) & MASK for value in (0, 0x7fffffff, -0x80000000, -1)]
    np.testing.assert_array_equal(result.row_hashes, expected)
    assert int(result.row_hashes[-1]) != (MULTIPLIER * MULTIPLIER * 0xffffffff) & MASK
    bins = np.array([[0, 1, 2, 255], [1, 1, 2, 255]], np.uint8)
    predicates = group_projection(np.empty((0, 4), np.uint32), bins, component_types=[1, 2],
                                   component_features=[0, 1], component_thresholds=[1, 255])
    np.testing.assert_array_equal(predicates.row_hashes,
        _reference(np.empty((0, 4), np.uint32), bins, [1, 2], [0, 1], [1, 255]))


@METAL
def test_compound_gpu_ctr_table_predicts_original_catboost_rows(tmp_path, monkeypatch):
    from catboost import CatBoost, CatBoostRegressor
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._ctrs import compute_ctr

    monkeypatch.setattr(CatBoost, "_fit", lambda *args, **kwargs: pytest.fail("No CPU training is allowed"))
    rows = [["foo", 0., "x"], ["foo", 1., "x"], ["bar", 0., "y"], ["bar", 1., "y"]] * 4
    hashes = np.array([cat_feature_hashes([row[column] for row in rows]) for column in (0, 2)])
    # One-hot bin zero means foo. The same original category is also a tensor
    # component, testing exact signed extension plus both predicate types.
    bins = np.array([[int(row[1] > .5) for row in rows], [int(row[0] != "foo") for row in rows]], np.uint8)
    permutation = np.array([15, 0, 10, 7, 3, 5, 2, 12, 1, 14, 8, 13, 6, 9, 4, 11], np.uint32)
    groups = group_projection(hashes, bins, component_types=[0, 0, 1, 2],
                               component_features=[0, 1, 0, 1], component_thresholds=[0, 0, 0, 0],
                               permutation=permutation)
    targets = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1], np.float32)
    ctr = compute_ctr(groups.row_bins, targets, permutation=permutation)
    history = {}
    for row in permutation:
        key = tuple(rows[row])
        total, count = history.get(key, (0, 0))
        assert ctr.values[row] == pytest.approx((total + .5) / (count + 1), abs=1e-7)
        history[key] = total + int(targets[row]), count + 1
    foo_hash = int(cat_feature_hashes(["foo"])[0])
    foo_signed = foo_hash if foo_hash < 1 << 31 else foo_hash - (1 << 32)
    elements = [{"combination_element": "cat_feature_value", "cat_feature_index": 0},
                {"combination_element": "cat_feature_value", "cat_feature_index": 1},
                {"combination_element": "float_feature", "float_feature_index": 0, "border": .5},
                {"combination_element": "cat_feature_exact_value", "cat_feature_index": 0, "value": foo_signed}]
    identifier = json.dumps({"identifier": elements, "type": "Borders"}, separators=(",", ":"))
    borders = [.3, .5, .7]
    descriptor = {"identifier": identifier, "elements": elements, "ctr_type": "Borders", "borders": borders,
                  "prior_numerator": .5, "prior_denomerator": 1, "shift": 0, "scale": 1, "target_border_idx": 0}
    table = []
    for category, count, total in zip(ctr.hashes, ctr.counts, ctr.sums):
        table.extend([str(int(groups.unique_hashes[category])), int(count - total), int(total)])
    model = {"features_info": {
        "float_features": [{"feature_index": 0, "flat_feature_index": 1, "has_nans": False,
                            "nan_value_treatment": "AsIs", "borders": [.5]}],
        "categorical_features": [{"feature_index": 0, "flat_feature_index": 0, "values": [foo_signed]},
                                 {"feature_index": 1, "flat_feature_index": 2}], "ctrs": [descriptor]},
        "ctr_data": {identifier: {"hash_stride": 3, "hash_map": table, "counter_denominator": 0}},
        "oblivious_trees": [{"splits": [{"split_type": "OnlineCtr", "border": border,
                            "ctr_target_border_idx": 0, "split_index": index + 2}],
                             "leaf_values": [0, 2 ** index]} for index, border in enumerate(borders)],
        "scale_and_bias": [1., [0.]]}
    path = tmp_path / "gpu-compound-ctr.json"
    path.write_text(json.dumps(model))
    standard = CatBoostRegressor().load_model(str(path), format="json")
    heldout = rows[:4] + [["foo", 1., "unseen"], ["unknown", 0., "x"]]
    expected = []
    for row in heldout:
        total, count = history.get(tuple(row), (0, 0))
        value = np.float32((total + .5) / (count + 1))
        expected.append(sum(2 ** index for index, border in enumerate(borders) if value > np.float32(border)))
    np.testing.assert_array_equal(standard.predict(heldout), expected)


@pytest.mark.parametrize("kwargs", [
    {"cat_hashes": [[-1]]}, {"cat_hashes": [[1.5]]}, {"cat_hashes": [1]},
    {"cat_hashes": [[1]], "bins": [[256]]}, {"cat_hashes": [[1]], "bins": [[0, 1]]},
    {"cat_hashes": [[1]], "component_types": []},
    {"cat_hashes": [[1]], "component_types": [3]},
    {"cat_hashes": [[1]], "component_features": [1]},
    {"cat_hashes": [[1]], "component_types": [1], "component_features": [0]},
    {"cat_hashes": [[1]], "bins": [[0]], "component_types": [2, 0], "component_features": [0, 0]},
    {"cat_hashes": [[1]], "bins": [[0]], "component_types": [1], "component_thresholds": [256]},
    {"cat_hashes": [[1, 2]], "permutation": [0, 0]},
    {"cat_hashes": [[1, 2]], "permutation": [0, 2]},
])
def test_invalid_projection_inputs_reject_before_gpu(kwargs, monkeypatch):
    import catboost_metal._projection as module
    monkeypatch.setattr(module, "_build_library", lambda: pytest.fail("Invalid projection reached native code"))
    with pytest.raises(ValueError):
        group_projection(**kwargs)


@METAL
def test_c_api_rejects_output_overlap_and_invalid_history():
    library = _load(_build_library())
    params = _Params(2, 1, 0, 1)
    cats, feature, threshold = (ct.c_uint32 * 2)(0, 1), ct.c_uint32(0), ct.c_uint32(0)
    kind, stats, error = ct.c_uint8(0), _Stats(), ct.create_string_buffer(512)
    output, sorted_output, indices = (ct.c_uint64 * 2)(), (ct.c_uint64 * 2)(), (ct.c_uint32 * 2)()
    args = [ct.byref(params), cats, None, ct.byref(kind), ct.byref(feature), ct.byref(threshold),
            None, output, output, indices, ct.byref(stats), error, len(error)]
    assert library.cbm_projection_group(*args) == 1
    assert b"overlap" in error.value
    args[8] = sorted_output
    args[6] = (ct.c_uint32 * 2)(1, 1)
    assert library.cbm_projection_group(*args) == 1
    assert b"permutation" in error.value
