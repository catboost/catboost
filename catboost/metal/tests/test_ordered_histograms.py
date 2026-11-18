"""Partitioned Ordered histograms against independent occurrence sums.

CUDA's TComputeHistogramsHelper retains parent histograms between depths;
TPointwisePartOffsetsHelper selects the smaller sibling for direct work. These
tests check the resulting numerical partitions, without copying that algorithm.
The Metal probe uses the production histogram kernels and accepts explicit
sampled derivatives, so prefix/quality offsets can be checked independently of
the objective and bootstrap kernels. No CPU CatBoost fitting is used.
"""

import ctypes as ct
import hashlib
from pathlib import Path
import subprocess

import numpy as np
import pytest

from catboost_metal import _ordered
from cuda_ordered_reference import numeric_folds, train_reference
from test_ordered_training import apple_silicon, prohibit_cpu_training, dataset, options, compare


class HistogramParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "folds", "leaves", "candidates", "packed_rows", "reuse", "candidate_batch")]
    _fields_ += [("tile_budget", ct.c_uint64)]
    _fields_ += [("histogram_max_leaves", ct.c_uint32), ("reserved", ct.c_uint32)]


@pytest.fixture(scope="module")
def ordered_histogram_probe():
    source = Path(__file__).with_name("ordered_histogram_probe.mm")
    root = source.parent.parent
    digest = hashlib.sha256(source.read_bytes())
    for name in ("metal_kernels.h", "metal_kernel_abi.h", "metal_additional_objective_kernels.h",
                 "metal_objective_kernels.h", "metal_deep_partition_kernels.h",
                 "metal_ordered_histogram_kernels.h", "metal_ordered_histogram_runtime.h"):
        digest.update((root / "native" / name).read_bytes())
    destination = root / ".build" / f"ordered_histogram_probe_{digest.hexdigest()[:16]}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        built = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                                "-framework", "Foundation", "-framework", "Metal", str(source),
                                "-o", str(destination)], capture_output=True, text=True)
        assert built.returncode == 0, built.stderr
    library = ct.CDLL(str(destination))
    library.cbm_ordered_histogram_probe.argtypes = [ct.POINTER(HistogramParams)] + [ct.c_void_p] * 8 + [ct.c_char_p, ct.c_uint32]
    library.cbm_ordered_histogram_probe.restype = ct.c_int

    class Probe:
        @staticmethod
        def run(bins, permutation, folds, sampled, leaf_ids, candidates, leaves, *, tile_budget=1 << 28,
                reuse=True, candidate_batch_size=0, histogram_max_leaves=0):
            bins = np.ascontiguousarray(bins, np.uint8)
            permutation, folds, leaf_ids, candidates = (
                np.ascontiguousarray(value, np.uint32) for value in (permutation, folds, leaf_ids, candidates))
            sampled = np.ascontiguousarray(sampled, np.float32)
            rows, features = bins.shape
            assert permutation.shape == leaf_ids.shape == (rows,)
            assert folds.shape[1] == 4 and candidates.shape[1] == 2 and sampled.shape[1] == 2
            params = HistogramParams(rows, features, len(folds), leaves, len(candidates), len(sampled),
                                     reuse, candidate_batch_size, tile_budget, histogram_max_leaves, 0)
            statistics = np.zeros((len(candidates), leaves, len(folds), 2, 4), np.float32)
            diagnostics = np.zeros(5, np.uint64)
            error = ct.create_string_buffer(4096)
            buffers = [bins, permutation, folds, sampled, leaf_ids, candidates, statistics, diagnostics]
            code = library.cbm_ordered_histogram_probe(ct.byref(params), *[x.ctypes.data for x in buffers], error, len(error))
            if code:
                raise RuntimeError(error.value.decode())
            return dict(statistics=statistics, tile_count=int(diagnostics[0]), histogram_jobs=int(diagnostics[1]),
                        reused_parents=int(diagnostics[2]), kernel_dispatches=int(diagnostics[3]),
                        fallback_levels=int(diagnostics[4]))
    return Probe


def direct_statistics(bins, permutation, folds, sampled, leaf_ids, candidates, leaves):
    """Float64 sums of stored float32 values, using original observation masks.

    Inputs use row-major bins and [G,W] sampled derivatives. Output axes are
    candidate, leaf, fold, split side, [estimate W,G, quality W,G]. Absolute
    descriptor cursor offsets deliberately need not start at zero.
    """
    bins = np.asarray(bins, np.uint8)
    sampled = np.asarray(sampled, np.float32).astype(np.float64)
    permutation, leaf_ids = np.asarray(permutation), np.asarray(leaf_ids)
    output = np.zeros((len(candidates), leaves, len(folds), 2, 4), np.float64)
    for fold_id, (estimate_end, quality_end, offset, _) in enumerate(folds):
        estimate_end, quality_end, offset = map(int, (estimate_end, quality_end, offset))
        rows = permutation[:quality_end]
        values = sampled[offset:offset + quality_end]
        for candidate, (feature, border) in enumerate(candidates):
            keys = 2 * leaf_ids[rows] + (bins[rows, feature] > border)
            for begin, end, statistic in ((0, estimate_end, 0), (estimate_end, quality_end, 2)):
                for component, source in ((statistic, 1), (statistic + 1, 0)):
                    output[candidate, :, fold_id, :, component] = np.bincount(
                        keys[begin:end], weights=values[begin:end, source], minlength=leaves * 2
                    ).reshape(leaves, 2)
    return output


def histogram_problem(rows=1027, depth=3, *, selected=1, seed=9351):
    """Sparse leaves, signed gradients, ragged folds and unselected sentinels."""
    rng = np.random.default_rng(seed)
    leaves = 1 << depth
    bins = rng.integers(0, 256, (rows, 5), dtype=np.uint8)
    bins[:, 0] = rng.choice([0, 1, 254, 255], rows)
    bins[:, 1] %= 8
    bins[:, 2] %= 11
    bins[:, 3] = rng.choice([0, 126, 127, 128, 129, 255], rows)
    bins[:4, 4] = [0, 1, 254, 255]
    permutation = rng.permutation(rows).astype(np.uint32)
    # Leave both siblings of one parent empty, and make another child carry
    # observations whose sampled structure weight is exactly zero.
    occupied = [leaf for leaf in range(leaves) if leaf % 4 != 3]
    leaf_ids = rng.choice(occupied, rows).astype(np.uint32)
    folds = []
    boundaries = numeric_folds(rows)
    packed_permutation = sum(end for _, end in boundaries)
    offset = selected * packed_permutation
    for prefix, end in boundaries:
        folds.append([prefix, end, offset, selected])
        offset += end
    folds = np.asarray(folds, np.uint32)
    # Finite, very large sentinels catch an accidental relative cursor offset.
    sampled = np.full((offset + 19, 2), [1e7, 1e6], np.float32)
    for prefix, end, start, _ in folds:
        order = permutation[:end]
        weight = (2.0 ** rng.integers(-3, 4, int(end))).astype(np.float32)
        weight[leaf_ids[order] % 4 == 2] = 0
        gradient = (rng.normal(size=int(end)).astype(np.float32) * weight).astype(np.float32)
        sampled[start:start + end] = np.column_stack((gradient, weight))
    candidates = np.asarray([
        [0, 254], [1, 0], [3, 127], [2, 8], [4, 254], [0, 0],
        [3, 126], [1, 6], [2, 3], [4, 0], [3, 128], [1, 1],
    ], np.uint32)
    return dict(bins=bins, permutation=permutation, folds=folds, sampled=sampled,
                leaf_ids=leaf_ids, candidates=candidates, leaves=leaves)


def assert_statistics(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=7e-6, atol=3e-5)
    assert np.isfinite(actual).all()
    # Weight lanes are masses; signed gradient lanes must stay signed.
    assert np.min(actual[..., (0, 2)]) >= -3e-5


@pytest.mark.parametrize("cache_leaves,batch", [(1, 1), (1, 5), (2, 2), (4, 11)])
def test_deeper_partitions_use_bounded_candidate_fallback(ordered_histogram_probe, cache_leaves, batch):
    data = histogram_problem(1031, 4)
    actual = ordered_histogram_probe.run(**data, histogram_max_leaves=cache_leaves,
                                         candidate_batch_size=batch, reuse=True)
    assert actual["fallback_levels"] > 0
    assert_statistics(actual["statistics"], direct_statistics(**data))


def test_depth16_dense_feature_does_not_reserve_an_unbounded_histogram():
    rows = 1031
    bins = np.asarray([np.where(np.arange(rows) % 2, 255, 0)], np.uint8)
    targets = np.where(bins[0], 1., -1.).astype(np.float32)
    features = np.zeros(255, np.uint32)
    borders = np.arange(255, dtype=np.uint32)
    config = options(iterations=1, depth=16, permutation_count=4)
    expected = train_reference(bins, targets, features, borders, **config)
    actual = _ordered.train(bins, targets, features, borders, **config)
    assert actual.depths[0] == 1
    compare(actual, expected)


@pytest.mark.parametrize("rows,depth", [(33, 0), (257, 3), (1031, 3), (8195, 4)])
@pytest.mark.parametrize("tile_budget", [4096, 1 << 28])
@pytest.mark.parametrize("reuse", [False, True])
def test_tiled_sibling_histograms_equal_direct_fold_occurrence_sums(
        ordered_histogram_probe, rows, depth, tile_budget, reuse):
    data = histogram_problem(rows, depth)
    expected = direct_statistics(**data)
    actual = ordered_histogram_probe.run(**data, tile_budget=tile_budget, reuse=reuse)
    assert_statistics(actual["statistics"], expected)
    assert actual["kernel_dispatches"] > 0 and actual["histogram_jobs"] > 0
    assert (actual["tile_count"] > 1) == (tile_budget == 4096)
    assert (actual["reused_parents"] > 0) == (reuse and depth > 0 and actual["tile_count"] == 1)
    if depth:
        # Entire empty parents and observed zero-weight children remain present
        # in the complete symmetric tree layout.
        np.testing.assert_allclose(actual["statistics"][:, 3::4], 0, atol=3e-5, rtol=0)
        np.testing.assert_allclose(actual["statistics"][:, 2::4], 0, atol=3e-5, rtol=0)


@pytest.mark.parametrize("reuse", [False, True])
def test_candidate_subsets_and_order_preserve_terminal_overflow_bucket(ordered_histogram_probe, reuse):
    data = histogram_problem(1031)
    full = ordered_histogram_probe.run(**data, tile_budget=4096, reuse=reuse)["statistics"]
    # Removing each feature's last threshold changes its compact histogram
    # span. Encoded values beyond that span must enter the overflow bucket.
    chosen = np.asarray([5, 8, 1, 6, 9, 11])
    subset = dict(data, candidates=data["candidates"][chosen])
    actual = ordered_histogram_probe.run(**subset, tile_budget=4096, reuse=reuse)["statistics"]
    assert_statistics(actual, direct_statistics(**subset))
    assert_statistics(actual, full[chosen])


@pytest.mark.parametrize("selected", [0, 2])
def test_histograms_conserve_split_sides_and_parent_children(ordered_histogram_probe, selected):
    data = histogram_problem(1031, depth=4, selected=selected)
    result = ordered_histogram_probe.run(**data, tile_budget=1 << 28, reuse=True)["statistics"]
    parent_data = dict(data, leaves=data["leaves"] // 2,
                       leaf_ids=data["leaf_ids"] & (data["leaves"] // 2 - 1))
    parents = direct_statistics(**parent_data)
    children_sum = result[:, :data["leaves"] // 2] + result[:, data["leaves"] // 2:]
    assert_statistics(children_sum, parents)
    # Summing split sides cannot depend on which candidate threshold was used.
    all_sides = result.sum(axis=3, dtype=np.float64)
    for candidate in range(1, len(data["candidates"])):
        assert_statistics(all_sides[candidate], all_sides[0])


@pytest.mark.parametrize("candidate_batch_size", [1, 2, 5, 11])
@pytest.mark.parametrize("tile_budget", [4096, 1 << 28])
def test_candidate_extraction_batches_keep_original_indices(ordered_histogram_probe, candidate_batch_size, tile_budget):
    data = histogram_problem(1031)
    actual = ordered_histogram_probe.run(**data, tile_budget=tile_budget, reuse=True,
                                        candidate_batch_size=candidate_batch_size)
    assert_statistics(actual["statistics"], direct_statistics(**data))


@pytest.mark.parametrize("rows", [8195, 32771])
@pytest.mark.parametrize("tile_budget", [4096, 1 << 28])
@pytest.mark.parametrize("reuse", [False, True])
def test_signed_cancellation_survives_tiles_prefix_scan_and_sibling_subtraction(
        ordered_histogram_probe, rows, tile_budget, reuse):
    data = histogram_problem(rows)
    data["permutation"] = np.arange(rows, dtype=np.uint32)
    # Each three-row group contributes +2**24, +1, -2**24 to one exact bin
    # and leaf. A float32 accumulator that loses its low part returns zero.
    # Group boundaries cross both histogram tiles and the 65536-occurrence
    # radix prefix boundary in the larger fixture.
    groups = np.arange(rows) // 3
    data["leaf_ids"] = np.asarray([0, 1, 2, 4, 5, 6], np.uint32)[groups % 6]
    for feature in range(data["bins"].shape[1]):
        data["bins"][:, feature] = ((groups * (feature * 2 + 1)) % 256).astype(np.uint8)
    base = 197
    data["folds"] = np.asarray([[123, rows, base, 2], [rows - rows % 3, rows, base + rows, 2]], np.uint32)
    data["sampled"] = np.full((base + 2 * rows + 19, 2), [1e7, 1e6], np.float32)
    gradient = np.asarray([2**24, 1, -(2**24)], np.float32)[np.arange(rows) % 3]
    for _, end, offset, _ in data["folds"]:
        data["sampled"][offset:offset + end, 0] = gradient
        data["sampled"][offset:offset + end, 1] = 1
    expected = direct_statistics(**data).astype(np.float32)
    actual = ordered_histogram_probe.run(**data, tile_budget=tile_budget, reuse=reuse)["statistics"]
    # All inputs and expected rounded totals are exactly representable; this
    # specifically requires retaining the small signed-gradient contribution.
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(expected[..., (1, 3)] == 1) > 0


@pytest.mark.parametrize("rows", [127, 1031])
@pytest.mark.parametrize("score_function", ["Cosine", "NewtonCosine"])
@pytest.mark.parametrize("normalize", [False, True])
def test_partitioned_training_preserves_weighted_prefix_and_full_estimation(
        rows, score_function, normalize):
    bins, targets, features, borders, weights = dataset("CrossEntropy", rows=rows, seed=2304)
    config = options("CrossEntropy", iterations=3, depth=4, score_function=score_function,
                     sample_weight=weights, permutation_count=4,
                     fold_size_loss_normalization=normalize, leaf_estimation_iterations=3)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)
    np.testing.assert_array_equal(state["descriptors"], expected["folds"])
    assert state["selection_rng"] == expected["selection_rng"]
    np.testing.assert_allclose(actual.leaf_weights.sum(axis=1), weights.sum(dtype=np.float64))


@pytest.mark.parametrize("permutation_count", [1, 4])
@pytest.mark.parametrize("border", [0, 254])
def test_terminal_bins_and_zero_weight_child_preserve_repeated_split_stop(permutation_count, border):
    # The only threshold separates bins 0 and 255. Zero-weight observations
    # retain their row partitions, while the one effective split must trigger
    # the CUDA repeated-winner stopping rule on the next attempted depth.
    bins = np.asarray([[0, 0, 255, 0, 255, 0, 255, 255, 255]], np.uint8)
    targets = (bins[0] > border).astype(np.float32)
    weights = np.asarray([0, 0, 2, 0, 0, 0, 4, 1, .5], np.float32)
    features, borders = np.asarray([0], np.uint32), np.asarray([border], np.uint32)
    config = options(iterations=3, depth=7, permutation_count=permutation_count,
                     sample_weight=weights, bias=0, l2_leaf_reg=.7)
    expected = train_reference(bins, targets, features, borders, **config)
    with _ordered.Session(bins, targets, features, borders, **config) as session:
        for _ in range(config["iterations"]):
            session.step()
        actual, state = session.result(), session.state()
    compare(actual, expected)
    np.testing.assert_array_equal(actual.depths, [1, 1, 1])
    np.testing.assert_array_equal(actual.leaf_weights[:, 0], 0)
    np.testing.assert_array_equal(actual.leaf_values[:, 0], 0)
    assert state["selection_rng"] == expected["selection_rng"]
    np.testing.assert_allclose(state["cursors"], expected["cursors"], rtol=8e-6, atol=3e-6)


def test_candidate_order_changes_do_not_change_a_strict_winning_structure():
    bins, targets, features, borders, weights = dataset(rows=1031, seed=4703)
    config = options(iterations=3, depth=4, sample_weight=weights, permutation_count=4)
    # Non-monotonic feature and border order catches treating a tile-local
    # index as the global candidate index used by the winner and repeat check.
    order = np.asarray([20, 0, 8, 15, 3, 19, 6, 9, 12, 5, 1, 18, 2, 14, 7, 16, 10, 4, 13, 17, 11])
    expected = train_reference(bins, targets, features, borders, **config)
    original = _ordered.train(bins, targets, features, borders, **config)
    reordered = _ordered.train(bins, targets, features[order], borders[order], **config)
    compare(original, expected)
    compare(reordered, expected)
    assert original.stats["search_permutations"] == reordered.stats["search_permutations"]
