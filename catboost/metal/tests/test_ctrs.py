"""Compare real Metal CTR scans with scalar CUDA history equations."""

import platform

import numpy as np
import pytest

from catboost_metal._ctrs import compute_ctr


METAL = pytest.mark.skipif(platform.system() != "Darwin" or platform.machine() != "arm64",
                          reason="CTR GPU tests require Apple Silicon")


def _reference(hashes, targets, permutation, ctr_type, border, numerator, denominator):
    history = {}
    output = np.empty(len(hashes), np.float64)
    for row in permutation:
        key = int(hashes[row])
        total, count = history.get(key, (0.0, 0))
        output[row] = (total + numerator) / (count + denominator)
        value = (float(targets[row] > border) if ctr_type == "Borders"
                 else float(targets[row] == border) if ctr_type == "Buckets"
                 else float(targets[row]) if ctr_type == "FloatTargetMeanValue" else 1.0)
        history[key] = total + value, count + 1
    if ctr_type == "FeatureFreq":
        output = np.asarray([(history[int(value)][1] + numerator) / (len(hashes) + denominator)
                             for value in hashes])
    return output, history


@METAL
@pytest.mark.parametrize("rows", [1, 1031, 2049])
@pytest.mark.parametrize("ctr_type", ["Borders", "Buckets", "FloatTargetMeanValue", "FeatureFreq"])
def test_gpu_ctr_matches_exclusive_reference(rows, ctr_type):
    rng = np.random.default_rng(201)
    hashes = rng.integers(0, 97, rows, dtype=np.uint32)
    targets = (rng.normal(size=rows) if ctr_type == "FloatTargetMeanValue"
               else rng.integers(0, 5, rows)).astype(np.float32)
    permutation = rng.permutation(rows)
    expected, history = _reference(hashes, targets, permutation, ctr_type, 2, 0.25, 2.0)
    result = compute_ctr(hashes, targets, ctr_type=ctr_type, target_border_idx=2,
                         permutation=permutation, prior_numerator=0.25, prior_denominator=2.0)
    assert result.stats["backend"] == "Metal"
    assert result.stats["category_sort_backend"] == "Metal"
    assert result.stats["kernel_dispatches"] >= 2
    np.testing.assert_allclose(result.values, expected, atol=2e-6, rtol=3e-6)
    for index, key in enumerate(result.hashes):
        expected_sum, expected_count = history[int(key)]
        np.testing.assert_allclose(result.sums[index], expected_sum, atol=2e-5, rtol=3e-6)
        assert result.counts[index] == expected_count
    full_values = result.full_values(np.r_[hashes, np.uint32(4294967295)])
    expected_full = []
    for key in np.r_[hashes, np.uint32(4294967295)]:
        total, count = history.get(int(key), (0.0, 0))
        expected_full.append((count + 0.25) / (rows + 2.0) if ctr_type == "FeatureFreq"
                             else (total + 0.25) / (count + 2.0))
    np.testing.assert_allclose(full_values, expected_full, atol=2e-6, rtol=3e-6)


@METAL
def test_current_and_future_targets_do_not_leak_into_training_ctr():
    hashes = np.array([1, 1, 1, 1, 2, 1], np.uint32)
    permutation = np.array([2, 4, 0, 5, 1, 3])
    targets = np.array([2, 3, 4, 5, 6, 7], np.float32)
    before = compute_ctr(hashes, targets, ctr_type="FloatTargetMeanValue", permutation=permutation)
    targets[permutation[3:]] = [10000, -6000, 9000]
    after = compute_ctr(hashes, targets, ctr_type="FloatTargetMeanValue", permutation=permutation)
    # Including the row whose target changed: its own value uses only earlier rows.
    np.testing.assert_array_equal(before.values[permutation[:4]], after.values[permutation[:4]])
    assert before.full_values([1])[0] != after.full_values([1])[0]


@METAL
def test_ctr_grouping_uses_gpu_sort_preserving_supplied_history(monkeypatch):
    hashes = np.array([0xffffffff, 7, 0x80000000, 7, 0xffffffff, 0, 7], np.uint32)
    targets = np.array([1, 0, 1, 1, 0, 1, 1], np.float32)
    permutation = np.array([4, 3, 5, 1, 2, 6, 0], np.uint32)
    expected, _ = _reference(hashes, targets, permutation, "Borders", 0, .5, 1)

    def forbidden(*args, **kwargs):
        raise AssertionError("CTR grouping must run on Metal, without NumPy sorting")

    with monkeypatch.context() as patch:
        for name in ("sort", "argsort", "unique", "lexsort"):
            patch.setattr(np, name, forbidden)
        result = compute_ctr(hashes, targets, permutation=permutation)
    np.testing.assert_allclose(result.values, expected, atol=1e-7)
    np.testing.assert_array_equal(result.hashes, [0, 7, 0x80000000, 0xffffffff])
    assert result.stats["category_sort"]["radix_passes"] == 8


@METAL
def test_large_single_segment_and_high_cardinality():
    rows = 65539
    hashes = np.repeat(np.arange(4097, dtype=np.uint32), 17)[:rows]
    targets = np.arange(rows, dtype=np.float32) % 2
    result = compute_ctr(hashes, targets, permutation=np.arange(rows - 1, -1, -1))
    assert len(result.hashes) > 255
    assert result.counts.sum(dtype=np.uint64) == rows
    np.testing.assert_allclose(result.values[-1], 0.5)
    singleton = compute_ctr(np.ones(rows, np.uint32), targets)
    assert singleton.counts[0] == rows
    np.testing.assert_allclose(singleton.values[-1], (targets[:-1].sum() + .5) / rows, atol=1e-7)


@METAL
@pytest.mark.parametrize("ctr_type", ["Borders", "Buckets", "FeatureFreq"])
def test_gpu_final_tables_match_standard_catboost_inference(tmp_path, monkeypatch, ctr_type):
    import json
    from catboost import CatBoost, CatBoostRegressor
    from catboost_metal._categorical import cat_feature_hashes
    from catboost_metal._ctr_model import ctr_feature_json, ctr_table_json, online_ctr_split_json

    def forbidden(*args, **kwargs):
        raise AssertionError("CTR checks must not run CPU training")
    monkeypatch.setattr(CatBoost, "_fit", forbidden)
    rows = ["foo", "bar", "foo", "hello", "foo", "bar", "hello", "foo"]
    targets = np.array([0, 0, 1, 1, 1, 0, 0, 1], np.float32)
    result = compute_ctr(cat_feature_hashes(rows), targets, ctr_type=ctr_type,
                         target_border_idx=1 if ctr_type == "Buckets" else 0,
                         permutation=[7, 3, 4, 1, 5, 0, 2, 6])
    borders = [0.2, 0.4, 0.6, 0.8]
    descriptor = ctr_feature_json(0, ctr_type, borders,
                                  target_border_idx=1 if ctr_type == "Buckets" else 0)
    table = ctr_table_json(result.hashes, result.counts, ctr_type=ctr_type,
                           **({} if ctr_type == "FeatureFreq" else {"sums": result.sums}))
    payload = {
        "features_info": {"categorical_features": [
            {"feature_index": 0, "flat_feature_index": 0, "feature_id": "category"}], "ctrs": [descriptor]},
        "ctr_data": {descriptor["identifier"]: table},
        "oblivious_trees": [{"splits": [online_ctr_split_json(border, index)],
                             "leaf_values": [0, 2 ** index], "leaf_weights": [1, 1]}
                            for index, border in enumerate(borders)],
        "scale_and_bias": [1.0, [0.0]],
    }
    path = tmp_path / "gpu-ctr-model.json"
    path.write_text(json.dumps(payload))
    model = CatBoostRegressor().load_model(str(path), format="json")
    heldout = ["foo", "bar", "hello", "unseen"]
    full = result.full_values(cat_feature_hashes(heldout))
    expected = sum((full > np.float32(border)) * 2 ** index for index, border in enumerate(borders))
    np.testing.assert_array_equal(model.predict([[value] for value in heldout]), expected)
    assert result.full_values([]).shape == (0,)


@pytest.mark.parametrize("kwargs", [
    {"category_hashes": [-1], "targets": [0]},
    {"category_hashes": [1.5], "targets": [0]},
    {"category_hashes": [], "targets": []},
    {"category_hashes": [1, 2], "targets": [0]},
    {"category_hashes": [1], "targets": [np.inf]},
    {"category_hashes": [1], "targets": [0.5]},
    {"category_hashes": [1], "targets": [-1]},
    {"category_hashes": [1], "targets": [256]},
    {"category_hashes": [1], "targets": [0], "ctr_type": "Counter"},
    {"category_hashes": [1], "targets": [0], "prior_denominator": 0},
    {"category_hashes": [1], "targets": [0], "prior_numerator": np.nan},
    {"category_hashes": [1], "targets": [0], "target_border_idx": -1},
    {"category_hashes": [1, 2], "targets": [0, 1], "permutation": [0, 0]},
    {"category_hashes": [1, 2], "targets": [0, 1], "permutation": [0, 2]},
])
def test_invalid_ctr_inputs_fail_before_native_call(kwargs):
    with pytest.raises(ValueError):
        compute_ctr(**kwargs)
