"""CUDA greedy-search equations and stable partitions on an actual Metal GPU."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class GreedyParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "leaves", "candidates", "total_bins", "score_groups",
        "score_function", "min_data_in_leaf", "max_depth", "max_leaves", "policy", "dimensions",
        "histogram_stride", "leaf_stride", "multiclass_optimization", "normalize")] + [
        ("l2", ct.c_float), ("feature_begin", ct.c_uint32),
        ("reserved0", ct.c_uint32), ("reserved1", ct.c_uint32)]


SPLIT_DTYPE = np.dtype([(name, "<u4") for name in ("index", "feature", "bin", "type")] + [
    ("gain", "<f4"), ("valid", "<u4"), ("error", "<u4"), ("leaf", "<u4")])
MISSING = np.iinfo(np.uint32).max


@pytest.fixture(scope="module")
def greedy_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("greedy_probe.mm")
    header = source.parent.parent / "native/metal_greedy_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:16]
    library_path = source.parent.parent / ".build" / f"greedy_probe_{digest}.dylib"
    library_path.parent.mkdir(exist_ok=True)
    if not library_path.exists():
        result = subprocess.run([
            "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(library_path)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    library = ct.CDLL(str(library_path))
    library.cbm_greedy_score_probe.argtypes = [ct.POINTER(GreedyParams)] + [ct.c_void_p] * 12 + [ct.c_uint32]
    library.cbm_greedy_frontier_probe.argtypes = [ct.POINTER(GreedyParams)] + [ct.c_void_p] * 14 + [ct.c_uint32]
    library.cbm_greedy_score_probe.restype = library.cbm_greedy_frontier_probe.restype = ct.c_int

    class Probe:
        def score(self, sums, weights, leaf_sums, leaf_weights, features, bins, types, offsets,
                  *, score=0, l2=3, groups=1, feature_weights=None, noise=None, missing_class=False,
                  normalize=False, feature_begin=0, padding=0):
            sums, weights, leaf_sums, leaf_weights = [np.ascontiguousarray(x, np.float32)
                for x in (sums, weights, leaf_sums, leaf_weights)]
            dimensions, leaves, total_bins = sums.shape
            features, bins, offsets = [np.ascontiguousarray(x, np.uint32) for x in (features, bins, offsets)]
            types = np.ascontiguousarray(types, np.uint8)
            global_features = max(int(features.max()) + 1, feature_begin + len(offsets) - 1)
            feature_weights = np.ascontiguousarray(np.ones(global_features) if feature_weights is None else feature_weights, np.float32)
            noise = np.ascontiguousarray(np.zeros(global_features) if noise is None else noise, np.float32)
            assert weights.shape == (leaves, total_bins) and leaf_sums.shape == (dimensions, leaves)
            assert leaf_weights.shape == (leaves,) and features.shape == bins.shape == types.shape
            assert len(feature_weights) == len(noise) == global_features
            histogram_stride, leaf_stride = leaves * total_bins + padding, leaves + padding
            padded_sums = np.full((dimensions, histogram_stride), np.nan, np.float32)
            padded_leaves = np.full((dimensions, leaf_stride), np.nan, np.float32)
            padded_sums[:, :leaves * total_bins] = sums.reshape(dimensions, -1)
            padded_leaves[:, :leaves] = leaf_sums
            params = GreedyParams(1, len(offsets) - 1, leaves, len(features), total_bins, groups, score, 1,
                16, leaves, 0, dimensions, histogram_stride, leaf_stride, missing_class, normalize,
                l2, feature_begin, 0, 0)
            winners = np.zeros(leaves, SPLIT_DTYPE)
            error = ct.create_string_buffer(4096)
            arrays = [padded_sums, weights, padded_leaves, leaf_weights, features, bins, types,
                      offsets, feature_weights, noise, winners]
            code = library.cbm_greedy_score_probe(ct.byref(params), *[x.ctypes.data for x in arrays], error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return winners

        def frontier(self, winners, offsets, depths, bins, indices, leaf_ids, *, policy=0,
                     min_data=1, max_depth=16, max_leaves=None):
            winners = np.ascontiguousarray(winners, SPLIT_DTYPE)
            offsets, depths, indices, leaf_ids = [np.ascontiguousarray(x, np.uint32)
                for x in (offsets, depths, indices, leaf_ids)]
            bins = np.ascontiguousarray(bins, np.uint8)
            leaves = len(winners)
            max_leaves = 2 * leaves if max_leaves is None else max_leaves
            params = GreedyParams(bins.shape[1], bins.shape[0], leaves, 1, 1, 1, 0, min_data,
                max_depth, max_leaves, policy, 1, leaves, leaves, 0, 0, 3, 0, 0, 0)
            selected, right_ids = np.zeros(leaves, np.uint32), np.zeros(leaves, np.uint32)
            new_leaf_ids, new_indices = np.zeros_like(leaf_ids), np.zeros_like(indices)
            new_offsets, new_depths = np.zeros(max_leaves + 1, np.uint32), np.zeros(max_leaves, np.uint32)
            info = np.zeros(4, np.uint32)
            error = ct.create_string_buffer(4096)
            arrays = [winners, offsets, depths, bins, indices, leaf_ids, selected, right_ids,
                      new_leaf_ids, new_indices, new_offsets, new_depths, info]
            code = library.cbm_greedy_frontier_probe(ct.byref(params), *[x.ctypes.data for x in arrays], error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return {"selected": selected[:info[0]], "right": right_ids, "ids": new_leaf_ids,
                    "indices": new_indices, "offsets": new_offsets[:info[1] + 1],
                    "depths": new_depths[:info[1]], "error": info[2]}
    return Probe()


def histogram_fixture(seed=705, dimensions=1):
    """Build sufficient statistics independently from weighted observations."""
    rng = np.random.default_rng(seed)
    sizes = [19, 53, 257, 17, 37]
    leaf_ids = np.repeat(np.arange(len(sizes)), sizes)
    rows = len(leaf_ids)
    raw_weights = rng.uniform(.02, 4, rows)
    raw_weights[::5] = 0
    gradients = rng.normal(0, 3, (dimensions, rows)) * raw_weights
    border_sizes = [3, 5, 2, 7]
    offsets = np.r_[0, np.cumsum(border_sizes)].astype(np.uint32)
    matrix = np.array([rng.integers(0, size, rows) for size in border_sizes])
    histogram = np.zeros((dimensions, len(sizes), offsets[-1]), np.float32)
    weights = np.zeros((len(sizes), offsets[-1]), np.float32)
    leaf_sums = np.zeros((dimensions, len(sizes)), np.float32)
    leaf_weights = np.zeros(len(sizes), np.float32)
    for leaf in range(len(sizes)):
        mask = leaf_ids == leaf
        leaf_sums[:, leaf] = gradients[:, mask].sum(axis=1)
        leaf_weights[leaf] = raw_weights[mask].sum()
        for feature, size in enumerate(border_sizes):
            for border in range(size):
                selected = mask & (matrix[feature] == border if feature == 1 else matrix[feature] <= border)
                histogram[:, leaf, offsets[feature] + border] = gradients[:, selected].sum(axis=1)
                weights[leaf, offsets[feature] + border] = raw_weights[selected].sum()
    candidates = [(f, b, int(f == 1)) for f, size in enumerate(border_sizes)
                  for b in range(size if f == 1 else size - 1)]
    features, bins, types = np.array(candidates, np.uint32).T
    return histogram, weights, leaf_sums, leaf_weights, features, bins, types, offsets


def gain_reference(args, *, score=0, l2=3, feature_weights=None, missing_class=False, normalize=False):
    """Float64 score equations from float32 statistics, not a training engine."""
    sums, weights, leaf_sums, leaf_weights, features, bins, _, offsets = args
    sums, weights, leaf_sums, leaf_weights = [np.asarray(x, np.float32).astype(np.float64)
        for x in (sums, weights, leaf_sums, leaf_weights)]
    feature_weights = np.ones(len(offsets) - 1) if feature_weights is None else np.asarray(feature_weights, np.float32)
    result = np.zeros((len(leaf_weights), len(features)))

    def quality(gradient, denominator):
        if score == 4:
            return -sum(g * g * (1 + 2 * np.log1p(w)) / w if w > 1e-20 else 0
                        for g, w in zip(gradient, denominator))
        if score == 5:
            return -sum(g * g / w * (w / (w - 1))**2 if w > 1 else 0
                        for g, w in zip(gradient, denominator))
        if score == 6:
            return -sum(g * g / w * (w * (w - 2) / (w * w - 3 * w + 1)) if w > 2 else 0
                        for g, w in zip(gradient, denominator))
        if score % 2 == 0:
            return -sum(g * g / (w + l2) if w > 1e-20 else 0 for g, w in zip(gradient, denominator))
        estimates = [g / (w + (l2 * w if normalize else l2)) if w > 0 else 0
                     for g, w in zip(gradient, denominator)]
        return -np.dot(gradient, estimates) / np.sqrt(1e-10 + np.dot(denominator, np.square(estimates)))

    for leaf in range(len(leaf_weights)):
        for candidate, (feature, border) in enumerate(zip(features, bins)):
            cell = offsets[feature] + border
            left_weight = max(weights[leaf, cell], 0)
            right_weight = max(leaf_weights[leaf] - left_weight, 0)
            if min(left_weight, right_weight) < 1e-20:
                continue
            parent = leaf_sums[:, leaf]
            left = sums[:, leaf, cell]
            if missing_class:
                parent, left = np.r_[parent, -parent.sum()], np.r_[left, -left.sum()]
            after = quality(np.column_stack([left, parent - left]).ravel(),
                            np.tile([left_weight, right_weight], len(left)))
            before = quality(parent, np.full(len(parent), leaf_weights[leaf]))
            result[leaf, candidate] = (after - before) * feature_weights[feature]
    return result


@pytest.mark.parametrize("score", [0, 1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("l2", [0, .5, 11])
def test_weighted_per_leaf_gains(greedy_probe, score, l2):
    args = histogram_fixture()
    feature_weights = np.array([.5, 2, 0, 1.2], np.float32)
    actual = greedy_probe.score(*args, score=score, l2=l2, groups=3, feature_weights=feature_weights)
    expected = gain_reference(args, score=score, l2=l2, feature_weights=feature_weights)
    chosen = expected.argmin(axis=1)
    np.testing.assert_array_equal(actual["index"], chosen)
    np.testing.assert_allclose(actual["gain"], expected[np.arange(len(chosen)), chosen], rtol=3e-5, atol=2e-4)
    assert actual["valid"].all() and not actual["error"].any()


@pytest.mark.parametrize("dimensions,missing_class", [(2, False), (3, True), (7, True)])
@pytest.mark.parametrize("score,normalize", [(0, False), (1, False), (1, True), (4, False), (5, False), (6, False)])
def test_multidimensional_gains_with_padded_planes(greedy_probe, dimensions, missing_class, score, normalize):
    args = histogram_fixture(dimensions=dimensions)
    actual = greedy_probe.score(*args, score=score, normalize=normalize, missing_class=missing_class, padding=17)
    expected = gain_reference(args, score=score, normalize=normalize, missing_class=missing_class)
    chosen = expected.argmin(axis=1)
    np.testing.assert_array_equal(actual["index"], chosen)
    np.testing.assert_allclose(actual["gain"], expected[np.arange(len(chosen)), chosen], rtol=3e-5, atol=2e-4)


@pytest.mark.parametrize("groups", [1, 2, 7, 257])
def test_candidate_ties_across_blocks_choose_original_index(greedy_probe, groups):
    features = 601
    args = (np.tile(np.array([[[-5, 0]]], np.float32), (1, 3, features)),
            np.tile(np.array([[2, 4]], np.float32), (3, features)),
            np.zeros((1, 3), np.float32), np.full(3, 4, np.float32),
            np.arange(features, dtype=np.uint32), np.zeros(features, np.uint32),
            np.zeros(features, np.uint8), np.arange(features + 1, dtype=np.uint32) * 2)
    actual = greedy_probe.score(*args, groups=groups)
    np.testing.assert_array_equal(actual["index"], 0)
    np.testing.assert_allclose(actual["gain"], -10)


def test_feature_tile_preserves_global_candidate_ids(greedy_probe):
    args = histogram_fixture()
    full = greedy_probe.score(*args)
    tile_results = []
    for feature in range(len(args[-1]) - 1):
        begin, end = args[-1][feature:feature + 2]
        tiled = (args[0][:, :, begin:end], args[1][:, begin:end], *args[2:7], [0, end - begin])
        tile_results.append(greedy_probe.score(*tiled, feature_begin=feature))
    for leaf in range(len(full)):
        best = min((tile[leaf] for tile in tile_results), key=lambda x: (x["gain"], x["index"]))
        assert best == full[leaf]


def test_zero_weight_children_have_zero_gain_and_remain_defined(greedy_probe):
    args = (np.array([[[0, 7]]], np.float32), np.array([[0, 2]], np.float32),
            np.array([[7]], np.float32), np.array([2], np.float32),
            [0, 0], [0, 1], [0, 0], [0, 2])
    actual = greedy_probe.score(*args)
    assert actual[0]["valid"] == 1 and actual[0]["index"] == 0 and actual[0]["gain"] == 0


@pytest.mark.parametrize("score", [0, 1, 4, 5, 6])
def test_every_candidate_gain_matches_independent_equations(greedy_probe, score):
    args = histogram_fixture(dimensions=3)
    expected = gain_reference(args, score=score, missing_class=True)
    for candidate in range(len(args[4])):
        one = (*args[:4], *(value[candidate:candidate + 1] for value in args[4:7]), args[7])
        actual = greedy_probe.score(*one, score=score, missing_class=True)
        np.testing.assert_allclose(actual["gain"], expected[:, candidate], rtol=3e-5, atol=2e-4)


def test_regularization_can_make_a_valid_split_gain_positive(greedy_probe):
    args = (np.array([[[4, 8]]], np.float32), np.array([[1, 2]], np.float32),
            np.array([[8]], np.float32), np.array([2], np.float32), [0], [0], [0], [0, 2])
    actual = greedy_probe.score(*args, l2=3)
    assert actual[0]["valid"] == 1
    assert actual[0]["gain"] == pytest.approx(4.8, abs=2e-6)


def test_nonfinite_score_propagates_even_when_another_candidate_wins(greedy_probe):
    args = list(histogram_fixture())
    args[0] = args[0].copy()
    args[0][0, 0, 0] = np.float32(1e30)
    actual = greedy_probe.score(*args, groups=7)
    assert actual[0]["error"] == 1 and actual[0]["valid"] == 1
    assert np.isfinite(actual[0]["gain"])


def test_sat_l2_weight_boundary_pole_and_large_weight(greedy_probe):
    pole = np.float32((3 + np.sqrt(5)) / 2)
    weights = np.array([0., 1., 2., np.nextafter(np.float32(2), np.float32(3)),
        np.nextafter(pole, np.float32(0)), pole, np.nextafter(pole, np.float32(4)),
        3., 4., 1e20], np.float32)
    parents = weights + np.where(weights > 1e10, weights, 5).astype(np.float32)
    args = (np.ones((1, len(weights), 1), np.float32), weights[:, None],
        np.full((1, len(weights)), 3., np.float32), parents,
        np.array([0], np.uint32), np.array([0], np.uint32), np.array([0], np.uint8),
        np.array([0, 1], np.uint32))
    expected = gain_reference(args, score=6)[:, 0]
    actual = greedy_probe.score(*args, score=6, l2=0)
    np.testing.assert_allclose(actual["gain"], expected, rtol=2e-6, atol=2e-6)
    assert np.isfinite(actual["gain"]).all() and not actual["error"].any()
    np.testing.assert_array_equal(actual, greedy_probe.score(*args, score=6, l2=100.))


def frontier_fixture(sizes, gains=None, *, seed=845):
    rng = np.random.default_rng(seed)
    leaves, rows = len(sizes), sum(sizes)
    indices = rng.permutation(rows).astype(np.uint32)
    leaf_ids = np.empty(rows, np.uint32)
    leaf_ids[indices] = np.repeat(np.arange(leaves), sizes)
    bins = rng.integers(0, 5, (3, rows), dtype=np.uint8)
    winners = np.zeros(leaves, SPLIT_DTYPE)
    winners["index"] = np.arange(leaves)
    winners["feature"] = np.arange(leaves) % 3
    winners["bin"] = 2
    winners["type"] = np.arange(leaves) % 2
    winners["gain"] = -np.ones(leaves) if gains is None else gains
    winners["valid"] = 1
    winners["leaf"] = np.arange(leaves)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    depths = np.full(leaves, 2, np.uint32)
    return winners, offsets, depths, bins, indices, leaf_ids


def check_partition(args, actual, selected):
    winners, _, depths, bins, indices, old_ids = args
    np.testing.assert_array_equal(actual["selected"], selected)
    expected_ids = old_ids.copy()
    expected_depths = np.r_[depths, np.zeros(len(selected), np.uint32)]
    right = np.full(len(winners), MISSING, np.uint32)
    for new, parent in enumerate(selected, start=len(winners)):
        split = winners[parent]
        go_right = bins[split["feature"]] == split["bin"] if split["type"] else bins[split["feature"]] > split["bin"]
        expected_ids[(old_ids == parent) & go_right] = new
        right[parent] = new
        expected_depths[parent] += 1
        expected_depths[new] = expected_depths[parent]
    expected_indices = indices[np.argsort(expected_ids[indices], kind="stable")]
    counts = np.bincount(expected_ids, minlength=len(winners) + len(selected))
    np.testing.assert_array_equal(actual["right"], right)
    np.testing.assert_array_equal(actual["ids"], expected_ids)
    np.testing.assert_array_equal(actual["indices"], expected_indices)
    np.testing.assert_array_equal(actual["offsets"], np.r_[0, np.cumsum(counts)])
    np.testing.assert_array_equal(actual["depths"], expected_depths)


@pytest.mark.parametrize("policy,selected", [(0, [0, 1, 3]), (1, [0]), (2, [1])])
def test_policy_frontier_ties_and_routing(greedy_probe, policy, selected):
    args = list(frontier_fixture([259, 31, 17, 513, 7], [-4, -4, .5, -2, 0]))
    args[2] = np.array([1, 3, 2, 3, 1], np.uint32)
    actual = greedy_probe.frontier(*args, policy=policy)
    check_partition(args, actual, selected)


@pytest.mark.parametrize("policy,selected", [(0, []), (1, [0]), (2, [])])
def test_non_improving_gain_selection_is_policy_specific(greedy_probe, policy, selected):
    args = frontier_fixture([19, 17], [0, .5])
    actual = greedy_probe.frontier(*args, policy=policy)
    check_partition(args, actual, selected)


@pytest.mark.parametrize("gain", [0, 3])
def test_region_root_can_split_without_improvement(greedy_probe, gain):
    args = list(frontier_fixture([7], [gain]))
    args[2][:] = 0
    actual = greedy_probe.frontier(*args, policy=2, min_data=100)
    check_partition(args, actual, [0])


def test_parent_count_and_depth_limits_not_child_count_or_weight(greedy_probe):
    args = list(frontier_fixture([5, 6, 7, 8], [-10, -9, -8, -7]))
    args[2] = np.array([1, 1, 4, 2], np.uint32)
    # Parent 1 has six rows, but its right child has only one. It may split.
    args[3][args[0][1]["feature"], args[4][5:11]] = 0
    args[3][args[0][1]["feature"], args[4][5]] = 2
    actual = greedy_probe.frontier(*args, min_data=5, max_depth=4)
    check_partition(args, actual, [1, 3])
    assert actual["offsets"][5] - actual["offsets"][4] == 1


@pytest.mark.parametrize("sizes", [[0, 1, 2, 255, 256, 257, 1031, 8193], [4] * 513,
                                  [131073], [0, 65535, 1, 0, 257, 0]])
def test_stable_variable_leaf_partitions_across_scan_boundaries(greedy_probe, sizes):
    args = frontier_fixture(sizes)
    actual = greedy_probe.frontier(*args, min_data=0)
    check_partition(args, actual, [i for i, size in enumerate(sizes) if size])


@pytest.mark.parametrize("split_type", [0, 1])
def test_routing_preserves_all_eight_bin_bits(greedy_probe, split_type):
    args = list(frontier_fixture([1031]))
    args[2][:] = 0
    args[0][0]["type"] = split_type
    args[0][0]["bin"] = 255 if split_type else 127
    args[3][0] = np.resize(np.array([0, 127, 128, 255], np.uint8), 1031)
    actual = greedy_probe.frontier(*args)
    check_partition(args, actual, [0])


def test_repeated_growth_retains_variable_leaf_ids_and_stable_offsets(greedy_probe):
    args = list(frontier_fixture([8193]))
    args[2][:] = 0
    for level in range(6):
        sizes = np.diff(args[1])
        chosen = int(np.argmax(sizes))
        args[0]["gain"] = -sizes.astype(np.float32)
        args[0]["feature"] = level % 3
        args[0]["bin"] = level % 4
        args[0]["type"] = level % 2
        actual = greedy_probe.frontier(*args, policy=1, min_data=0)
        check_partition(args, actual, [chosen])
        count = len(actual["depths"])
        winners = np.zeros(count, SPLIT_DTYPE)
        winners["valid"] = 1
        winners["leaf"] = np.arange(count)
        args = [winners, actual["offsets"], actual["depths"], args[3], actual["indices"], actual["ids"]]


def test_capacity_and_error_flags_prevent_extra_growth(greedy_probe):
    args = list(frontier_fixture([19, 21, 23, 25]))
    actual = greedy_probe.frontier(*args, max_leaves=5)
    check_partition(args, actual, [0])
    actual = greedy_probe.frontier(*args, max_leaves=4)
    check_partition(args, actual, [])
    args[0][2]["error"] = 1
    actual = greedy_probe.frontier(*args)
    assert actual["error"] == 1
    check_partition(args, actual, [])


@pytest.mark.parametrize("corruption", ["indices", "offsets", "winner", "depth"])
def test_probe_rejects_malformed_partition_metadata(greedy_probe, corruption):
    args = list(frontier_fixture([9, 11]))
    if corruption == "indices": args[4][0] = args[4][1]
    if corruption == "offsets": args[1][1] = 99
    if corruption == "winner": args[0][0]["feature"] = 3
    if corruption == "depth": args[2][0] = 17
    with pytest.raises(ValueError):
        greedy_probe.frontier(*args)
