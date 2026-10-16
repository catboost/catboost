"""Read-only pair access preserves stored order, weights, and Pool row mapping."""

import os

import numpy as np
import pytest
from catboost import Pool


pytestmark = pytest.mark.skipif(
    os.environ.get("CATBOOST_NATIVE_METAL_TESTS") != "1",
    reason="requires the rebuilt native CatBoost Pool pair accessors",
)

PAIRS = [[2, 0], [4, 5], [1, 2], [5, 3]]
WEIGHTS = [0.25, 2.5, 0.0, 0.75]


def flat_pool(explicit_weights=True):
    return Pool(np.arange(12, dtype=np.float32).reshape(6, 2),
                group_id=[0, 0, 0, 1, 1, 1], pairs=PAIRS,
                pairs_weight=WEIGHTS if explicit_weights else None)


def grouped_pool(tmp_path, scheme, explicit_weights=True):
    data = tmp_path / "groups.tsv"
    data.write_text("first\t0\nfirst\t1\nfirst\t2\nsecond\t3\nsecond\t4\nsecond\t5\n")
    columns = tmp_path / "groups.cd"
    columns.write_text("0\tGroupId\n1\tNum\n")
    pairs = tmp_path / "groups.pairs"
    groups = ["first", "second", "first", "second"] if scheme == "dsv-grouped" else ["0", "1", "0", "1"]
    local_pairs = [[2, 0], [1, 2], [1, 2], [2, 0]]
    lines = []
    for group, pair, weight in zip(groups, local_pairs, WEIGHTS):
        fields = [group, str(pair[0]), str(pair[1])]
        if explicit_weights:
            fields.append(str(weight))
        lines.append("\t".join(fields))
    pairs.write_text("\n".join(lines) + "\n")
    pair_path = scheme + "://" + str(pairs)
    return Pool(str(data), column_description=str(columns), pairs=pair_path), pair_path


def assert_pairs(pool, pairs, weights):
    result = pool.get_pairs()
    assert isinstance(result, list)
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in result)
    assert result == pairs
    assert isinstance(pool.get_pairs_weight(), list)
    np.testing.assert_array_equal(pool.get_pairs_weight(), np.asarray(weights, dtype=np.float32))
    assert len(result) == pool.num_pairs()


@pytest.mark.parametrize("explicit_weights", [False, True])
def test_flat_pairs_preserve_order_and_original_weights(explicit_weights):
    pool = flat_pool(explicit_weights)
    expected_weights = WEIGHTS if explicit_weights else [1.0] * len(PAIRS)
    assert_pairs(pool, PAIRS, expected_weights)
    # Returned objects are copies; callers cannot mutate the Pool through them.
    pairs = pool.get_pairs()
    weights = pool.get_pairs_weight()
    pairs[0][0] = 99
    weights[0] = 99
    assert_pairs(pool, PAIRS, expected_weights)
    pool.set_pairs_weight([3, 2, 1, 0])
    assert_pairs(pool, PAIRS, [3, 2, 1, 0])


def test_pool_without_pairs_returns_empty_lists():
    pool = Pool(np.arange(8, dtype=np.float32).reshape(4, 2))
    assert_pairs(pool, [], [])


def test_flat_pairs_without_groups_preserve_duplicates_and_zero_weight():
    pairs = [[2, 0], [1, 3], [2, 0]]
    weights = [0.0, 0.125, 2.0]
    pool = Pool(np.arange(8, dtype=np.float32).reshape(4, 2), pairs=pairs, pairs_weight=weights)
    assert_pairs(pool, pairs, weights)


def test_pair_weights_remain_separate_from_object_and_group_weights():
    pool = flat_pool()
    pool.set_group_weight([2.0, 2.0, 2.0, 0.5, 0.5, 0.5])
    pool.set_weight([1, 2, 3, 4, 5, 6])
    assert_pairs(pool, PAIRS, WEIGHTS)


@pytest.mark.parametrize("quantized", [False, True])
def test_flat_pair_slices_remap_indices_and_preserve_weights(quantized):
    pool = flat_pool()
    if quantized:
        pool.quantize(border_count=4)
    assert_pairs(pool.slice([3, 4, 5]), [[1, 2], [2, 0]], [2.5, 0.75])
    assert_pairs(pool.slice([3, 4, 5, 0, 1, 2]), [[5, 3], [1, 2], [4, 5], [2, 0]], WEIGHTS)


@pytest.mark.parametrize("scheme", ["dsv-grouped", "dsv-grouped-with-idx"])
@pytest.mark.parametrize("explicit_weights", [False, True])
def test_grouped_pair_storage_expands_to_current_pool_rows(tmp_path, scheme, explicit_weights):
    pool, _ = grouped_pool(tmp_path, scheme, explicit_weights)
    expected = WEIGHTS if explicit_weights else [1.0] * len(PAIRS)
    assert_pairs(pool, PAIRS, expected)
    assert_pairs(pool.slice([3, 4, 5]), [[1, 2], [2, 0]], [expected[1], expected[3]])
    assert_pairs(pool.slice([3, 4, 5, 0, 1, 2]), [[5, 3], [1, 2], [4, 5], [2, 0]], expected)
    pool.quantize(border_count=4)
    assert_pairs(pool, PAIRS, expected)
    assert_pairs(pool.slice([3, 4, 5]), [[1, 2], [2, 0]], [expected[1], expected[3]])


@pytest.mark.parametrize("storage", ["flat", "dsv-grouped", "dsv-grouped-with-idx"])
def test_quantized_pool_reload_reads_external_pair_storage(tmp_path, storage):
    if storage == "flat":
        pool = flat_pool()
        pairs = tmp_path / "flat.pairs"
        pairs.write_text("".join(f"{winner}\t{loser}\t{weight}\n" for (winner, loser), weight in zip(PAIRS, WEIGHTS)))
        pair_path = str(pairs)
    else:
        pool, pair_path = grouped_pool(tmp_path, storage)
    pool.quantize(border_count=4)
    path = tmp_path / "pairs.quantized"
    pool.save(path)
    # CatBoost's quantized data file uses external pairs; Pool.save does not
    # embed them. Supply the same pair file when reopening the quantized data.
    restored = Pool("quantized://" + str(path), pairs=pair_path)
    assert_pairs(restored, PAIRS, WEIGHTS)
    assert_pairs(restored.slice([3, 4, 5]), [[1, 2], [2, 0]], [2.5, 0.75])
    assert_pairs(Pool("quantized://" + str(path)), [], [])
