"""Real GPU checks for CUDA CtrHistoryUnit=Group, without model training."""

import ctypes as ct
import json
import platform

import numpy as np
import pytest

from catboost_metal import _ctrs
from catboost_metal._ctrs import compute_ctr


METAL = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                          reason="CTR GPU tests require Apple Silicon")
TYPES = ["Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq"]


def _reference(hashes, targets, order, groups, kind, border, numerator, denominator):
    history = {}
    output = np.empty(len(hashes), np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and groups[order[end]] == groups[order[start]]:
            end += 1
        for row in order[start:end]:
            total, count = history.get(int(hashes[row]), (0.0, 0))
            output[row] = (total + numerator) / (count + denominator)
        for row in order[start:end]:
            key = int(hashes[row])
            total, count = history.get(key, (0.0, 0))
            value = (float(targets[row] > border) if kind == "Borders"
                     else float(targets[row] == border) if kind == "Buckets"
                     else float(targets[row]) if kind == "FloatTargetMeanValue" else 1.0)
            history[key] = total + value, count + 1
        start = end
    if kind == "FeatureFreq":
        output = np.asarray([(history[int(value)][1] + numerator) / (len(hashes) + denominator)
                             for value in hashes])
    return output, history


@METAL
@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("block_order", [[0, 1, 2, 3], [2, 0, 3, 1]])
def test_unequal_shuffled_groups_match_reference_and_full_tables(kind, block_order):
    groups = np.repeat(np.array([0xffffffff, 7, 0, 0x80000000], np.uint32), [3, 1, 5, 2])
    blocks = np.split(np.arange(11, dtype=np.uint32), [3, 4, 9])
    order = np.concatenate([blocks[index] for index in block_order])
    hashes = np.array([3, 3, 8, 3, 8, 3, 8, 5, 3, 5, 8], np.uint32)
    targets = np.array([0, 2, 1, 3, 1, 0, 2, 3, 1, 0, 2], np.float32)
    if kind == "FloatTargetMeanValue":
        targets = targets * .125 - .25
    options = dict(ctr_type=kind, permutation=order, target_border_idx=1,
                   prior_numerator=.25, prior_denominator=2)
    result = compute_ctr(hashes, targets, group_ids=groups, **options)
    legacy = compute_ctr(hashes, targets, **options)
    expected, history = _reference(hashes, targets, order, groups, kind, 1, .25, 2)
    np.testing.assert_allclose(result.values, expected, atol=2e-6, rtol=2e-6)
    for name in ("hashes", "sums", "counts"):
        np.testing.assert_array_equal(getattr(result, name), getattr(legacy, name))
    assert result.stats["history_unit"] == "Group"
    if kind == "FeatureFreq":
        np.testing.assert_array_equal(result.values, legacy.values)
        assert result.stats["kernel_dispatches"] == legacy.stats["kernel_dispatches"]
    else:
        assert result.stats["kernel_dispatches"] == legacy.stats["kernel_dispatches"] + 1
    heldout = np.array([3, 5, 8, 123456], np.uint32)
    expected_full = []
    for key in heldout:
        total, count = history.get(int(key), (0.0, 0))
        expected_full.append((count + .25) / (len(hashes) + 2) if kind == "FeatureFreq"
                             else (total + .25) / (count + 2))
    np.testing.assert_allclose(result.full_values(heldout), expected_full, atol=1e-7)


@METAL
@pytest.mark.parametrize("kind", TYPES[:3])
def test_current_and_future_group_targets_cannot_change_prior_cursors(kind):
    groups = np.array([99, 99, 7, 7, 7, 7, 2, 2, 2], np.uint32)
    hashes = np.array([1, 2, 1, 1, 2, 1, 1, 2, 2], np.uint32)
    order = np.array([6, 7, 8, 1, 0, 4, 2, 5, 3], np.uint32)
    targets = np.array([0, 2, 1, 2, 3, 1, 1, 0, 2], np.float32)
    if kind == "FloatTargetMeanValue":
        targets = targets * .125 + .25
    options = dict(ctr_type=kind, target_border_idx=1, permutation=order, group_ids=groups)
    before = compute_ctr(hashes, targets, **options)
    targets[:6] = ([2**60, -2**59, 2**59, 2**58, -2**58, 2**60]
                   if kind == "FloatTargetMeanValue" else [2, 1, 3, 0, 1, 2])
    after = compute_ctr(hashes, targets, **options)
    # All rows of group 99, whose targets changed, still see exactly the prefix
    # before that group. Subtracting its huge sum from an inclusive sum fails.
    np.testing.assert_array_equal(before.values[order[:5]], after.values[order[:5]])
    assert not np.array_equal(before.sums, after.sums)


def _native_call(kind, groups, *, grouped=True, categories=None, indices=None):
    library = _ctrs._load(_ctrs._build_library())
    categories = np.array([0, 0, 1, 1] if categories is None else categories, np.uint32)
    indices = np.array([1, 3, 0, 2] if indices is None else indices, np.uint32)
    rows, count = len(indices), int(categories[-1]) + 1
    targets = np.array([0, 1, 2, 1], np.float32)[:rows]
    params = _ctrs._Params(rows, count, _ctrs.CTR_TYPES[kind], 0, .5, 1)
    values, sums = np.empty(rows, np.float32), np.empty(count, np.float32)
    counts = np.empty(count, np.uint32)
    stats, error = _ctrs._Stats(), ct.create_string_buffer(2048)
    u32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint32))
    f32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_float))
    args = [ct.byref(params), u32(categories), u32(indices), f32(targets)]
    if grouped:
        args.append(None if groups is None else u32(np.asarray(groups, np.uint32)))
    args.extend([f32(values), f32(sums), u32(counts), ct.byref(stats), error, len(error)])
    code = (library.cbm_compute_ctrs_grouped if grouped else library.cbm_compute_ctrs)(*args)
    return code, error.value, values, sums, counts, stats


@METAL
@pytest.mark.parametrize("kind", TYPES)
def test_singletons_and_null_group_pointer_preserve_legacy_behavior(kind):
    rng = np.random.default_rng(337)
    hashes = rng.integers(0, 67, 1031, dtype=np.uint32)
    targets = (rng.normal(size=1031) if kind == "FloatTargetMeanValue"
               else rng.integers(0, 4, 1031)).astype(np.float32)
    options = dict(ctr_type=kind, permutation=rng.permutation(1031))
    legacy = compute_ctr(hashes, targets, **options)
    singletons = compute_ctr(hashes, targets, group_ids=np.arange(1031, dtype=np.uint32), **options)
    for name in ("values", "sums", "counts"):
        np.testing.assert_array_equal(getattr(legacy, name), getattr(singletons, name))
    old = _native_call(kind, None, grouped=False)
    new = _native_call(kind, None)
    assert old[0] == new[0] == 0, (old[1], new[1])
    for index in (2, 3, 4):
        np.testing.assert_array_equal(old[index], new[index])
    assert old[5].kernel_dispatches == new[5].kernel_dispatches


@METAL
def test_large_category_segments_and_long_group_heads_use_gpu_ordering(monkeypatch):
    rows = 65539
    cuts = [0, 1, 5, 1029, 2048, 4099, 32770, rows]
    labels = [91, 0, 0x80000000, 17, 31, 0xffffffff, 23]
    groups = np.empty(rows, np.uint32)
    blocks = []
    for first, end, label in zip(cuts, cuts[1:], labels):
        groups[first:end] = label
        blocks.append(np.arange(first, end, dtype=np.uint32))
    order = np.concatenate([blocks[index] for index in [4, 0, 6, 2, 1, 5, 3]])
    rng = np.random.default_rng(203)
    hashes = rng.integers(0, 83, rows, dtype=np.uint32)
    targets = rng.normal(size=rows).astype(np.float32)
    expected, _ = _reference(hashes, targets, order, groups, "FloatTargetMeanValue", 0, .25, 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("Category/group ordering must run on Metal")

    with monkeypatch.context() as patch:
        for name in ("sort", "argsort", "lexsort", "unique"):
            patch.setattr(np, name, forbidden)
        result = compute_ctr(hashes, targets, ctr_type="FloatTargetMeanValue", permutation=order,
                             group_ids=groups, prior_numerator=.25, prior_denominator=2)
    np.testing.assert_allclose(result.values, expected, rtol=5e-6, atol=2e-6)
    assert result.counts.sum(dtype=np.uint64) == rows
    assert result.stats["category_sort_backend"] == "Metal"


@METAL
@pytest.mark.parametrize("kind", TYPES)
def test_grouped_full_tables_remain_standard_catboost_models(tmp_path, monkeypatch, kind):
    from catboost import CatBoost, CatBoostRegressor
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._ctr_model import ctr_feature_json, ctr_table_json, online_ctr_split_json

    def forbidden(*args, **kwargs):
        raise AssertionError("Grouped CTR checks must not run CPU training")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)
    rows = ["red", "blue", "red", "green", "blue", "red", "green", "blue"]
    targets = np.array([-2, 3, 1, 0, 5, -1, 2, 4] if kind == "FloatTargetMeanValue"
                       else [0, 1, 1, 0, 1, 0, 1, 1], np.float32)
    border = 1 if kind == "Buckets" else 0
    result = compute_ctr(cat_feature_hashes(rows), targets, ctr_type=kind, target_border_idx=border,
                         group_ids=[9, 9, 9, 3, 3, 1, 1, 1], permutation=[5, 7, 6, 0, 2, 1, 4, 3])
    borders = [-.5, .2, .5, 1.5]
    descriptor = ctr_feature_json(0, kind, borders, target_border_idx=border)
    table = ctr_table_json(result.hashes, result.counts, ctr_type=kind,
                           **({} if kind == "FeatureFreq" else {"sums": result.sums}))
    payload = {
        "features_info": {"categorical_features": [
            {"feature_index": 0, "flat_feature_index": 0, "feature_id": "category"}], "ctrs": [descriptor]},
        "ctr_data": {descriptor["identifier"]: table},
        "oblivious_trees": [{"splits": [online_ctr_split_json(value, index)],
                             "leaf_values": [0, 2 ** index], "leaf_weights": [1, 1]}
                            for index, value in enumerate(borders)],
        "scale_and_bias": [1.0, [0.0]],
    }
    path = tmp_path / "group-ctr.json"
    path.write_text(json.dumps(payload))
    model = CatBoostRegressor().load_model(str(path), format="json")
    heldout = ["red", "blue", "green", "unseen"]
    full = result.full_values(cat_feature_hashes(heldout))
    expected = sum((full > np.float32(value)) * 2 ** index for index, value in enumerate(borders))
    np.testing.assert_array_equal(model.predict([[value] for value in heldout]), expected)
    binary = tmp_path / "group-ctr.cbm"
    model.save_model(str(binary))
    restored = CatBoostRegressor().load_model(str(binary))
    np.testing.assert_array_equal(restored.predict([[value] for value in heldout]), expected)


@pytest.mark.parametrize("groups,order", [
    ([1], None), ([-1, 0, 0], None), ([0, 2**32, 0], None),
    ([False, True, True], None), ([0., 0., 1.], None),
    ([1, 2, 1], None), ([1, 1, 2], [0, 2, 1]),
])
def test_invalid_groups_rejected_before_gpu(monkeypatch, groups, order):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid group inputs must fail before GPU work")
    monkeypatch.setattr(_ctrs, "_build_library", forbidden)
    with pytest.raises(ValueError, match="group"):
        compute_ctr([1, 2, 1], [0, 1, 1], group_ids=groups, permutation=order)


@METAL
def test_native_group_reappearance_rejected():
    result = _native_call("FloatTargetMeanValue", [11, 22, 11], categories=[0, 0, 0], indices=[0, 1, 2])
    assert result[0] != 0
    assert b"contiguous" in result[1]
