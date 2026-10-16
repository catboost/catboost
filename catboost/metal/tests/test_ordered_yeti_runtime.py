"""Ordered YetiRank blocks, explicit seed packets and every prefix cursor.

The oracle calls the independent seeded pair reference separately for each
Learn/Quality or estimation block. It never fits a CatBoost model on CPU.
"""
import ctypes as ct
import math

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._native import YetiRankOptions, _f32, _u8, _u32
from cuda_ordered_group_reference import reference_group_folds
from test_ordered_query_runtime import (
    QueryRuntime, expected_descriptors, query_library, query_params, query_problem,
)
from test_ordered_training import apple_silicon, prohibit_cpu_training
from test_yeti_rank_kernels import reference


@pytest.fixture(scope="module")
def yeti_library(query_library):
    lib = query_library
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    u64 = ct.POINTER(ct.c_uint64)
    signatures = {
        "create_yeti_banked": [ct.POINTER(_ordered.Params), ct.c_uint32, ct.c_uint64, u8,
                              f32, f32, f32, u32, u32, u8, u32, ct.POINTER(YetiRankOptions),
                              u32, ct.c_double, ct.POINTER(ct.c_void_p)],
        "yeti_seed_shape": [ct.c_void_p, ct.c_uint32, u32, u32],
        "set_yeti_oracle_seeds": [ct.c_void_p, ct.c_uint32, u64],
        "set_yeti_leaf_seeds": [ct.c_void_p, ct.c_uint32, u64],
    }
    for name, args in signatures.items():
        operation = getattr(lib, "cbm_ordered_session_" + name)
        operation.argtypes, operation.restype = args + [ct.c_char_p, ct.c_size_t], ct.c_int
    return lib


class YetiRuntime(QueryRuntime):
    def __init__(self, lib, params, data, *, legacy=False, permutations=7, decay=.71):
        self.lib, self.params, self.data = lib, params, data
        self.handle = ct.c_void_p()
        self.options = YetiRankOptions(len(data["sizes"]), permutations, decay, int(legacy))
        cf = np.arange(params.candidates, dtype=np.uint32)
        cb, types = np.zeros(params.candidates, np.uint32), np.zeros(params.candidates, np.uint8)
        error = ct.create_string_buffer(4096)
        code = lib.cbm_ordered_session_create_yeti_banked(
            ct.byref(params), len(data["banks"]), data["banks"].size, _u8(data["banks"]),
            _f32(data["targets"]), _f32(data["weights"]), _f32(data["initial"]),
            _u32(cf), _u32(cb), _u8(types), _u32(data["permutations"]), ct.byref(self.options),
            _u32(data["offsets"]), data["growth"], ct.byref(self.handle), error, len(error))
        if code:
            assert not self.handle.value, "failed Yeti constructor retained a session"
            raise RuntimeError(error.value.decode())
        assert self.handle.value

    def seed_shape(self, selected):
        weak, leaf = ct.c_uint32(), ct.c_uint32()
        self.call("yeti_seed_shape", selected, ct.byref(weak), ct.byref(leaf))
        return weak.value, leaf.value

    def seeds(self, values, *, leaf=False):
        values = np.ascontiguousarray(values, np.uint64)
        self.call("set_yeti_leaf_seeds" if leaf else "set_yeti_oracle_seeds", len(values),
                  values.ctypes.data_as(ct.POINTER(ct.c_uint64)))


def local_offsets(data, order):
    """Infer complete query boundaries from original row identities."""
    order = np.asarray(order, np.int64)
    if not len(order):
        return np.array([0], np.uint32)
    groups = np.searchsorted(data["offsets"][1:], order, side="right")
    bounds = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(order)]
    for begin, end in zip(bounds[:-1], bounds[1:]):
        group = groups[begin]
        np.testing.assert_array_equal(order[begin:end], np.arange(data["offsets"][group], data["offsets"][group + 1]))
    return bounds.astype(np.uint32)


def active_tasks(data, descriptors):
    result = []
    for task, (prefix, _, _, bank) in enumerate(descriptors.astype(int)):
        sizes = np.diff(local_offsets(data, data["permutations"][bank, :prefix]))
        if task + 1 == len(descriptors) or np.any(sizes > 1):
            result.append(task)
    return result


def packet(data, p, descriptors, selected, iteration=0):
    folds = sum(int(task[3]) == selected for task in descriptors[:-1])
    estimates = len(active_tasks(data, descriptors))
    evaluations = p.leaf_iterations + int(p.leaf_iterations > 1)
    base = 0x8def012345670019 + iteration * 9049
    weak = np.asarray([base + 17 + index * 0x10000013 for index in range(2 * folds)], np.uint64)
    leaves = np.asarray([[base + 0xabcdef + evaluation * 2137 + task * 7919
                          for task in range(estimates)] for evaluation in range(evaluations)], np.uint64)
    return weak, leaves


def block_terms(data, order, point, seed, options):
    offsets = local_offsets(data, order)
    if not len(order):
        return np.empty((0, 2), np.float32)
    return reference(data["targets"][order], data["weights"][order], point, offsets,
                     seed=int(seed), permutations=options.permutations, decay=options.decay,
                     center_rows=len(offsets) - 1 if options.legacy_prefix_centering else None)[1]


def score_oracle(data, p, descriptors, cursors, selected, weak, options, *, object_weights=False):
    numerator, norm, position = 0., 1e-20, 0
    for prefix, quality, offset, bank in descriptors[:-1].astype(int):
        if bank != selected:
            continue
        order = data["permutations"][bank, :quality]
        point = cursors[offset:offset + quality]
        # Every block starts query/task IDs at zero and has its own oracle seed.
        learn = block_terms(data, order[:prefix], point[:prefix], weak[position], options)
        test = block_terms(data, order[prefix:], point[prefix:], weak[position + 1], options)
        position += 2
        children = data["banks"][bank, 0, order] > 0
        for side in (False, True):
            left, right = children[:prefix] == side, children[prefix:] == side
            mass = float(data["weights"][order[:prefix]][left].sum(dtype=float)) if object_weights else float(learn[left, 1].sum(dtype=float))
            quality_mass = float(data["weights"][order[prefix:]][right].sum(dtype=float)) if object_weights else float(test[right, 1].sum(dtype=float))
            ridge = p.l2 * (mass if p.normalize else 1.)
            value = float(learn[left, 0].sum(dtype=float)) / (mass + ridge) if mass > 0 else 0.
            numerator += float(test[right, 0].sum(dtype=float)) * value
            norm += quality_mass * value ** 2
    assert position == len(weak)
    assert norm > 1e-15, "the forced split must have nonzero quality curvature"
    return -numerator / math.sqrt(norm)


def leaf_oracle(data, p, descriptors, cursors, tree, seeds, options):
    leaves = 1 << tree["depth"]
    estimates = active_tasks(data, descriptors)
    assert seeds.shape == (p.leaf_iterations + int(p.leaf_iterations > 1), len(estimates))
    positions = {task: position for position, task in enumerate(estimates)}
    updated, all_values, all_weights = cursors.copy(), [], []
    for task, (prefix, quality, offset, bank) in enumerate(descriptors.astype(int)):
        order = data["permutations"][bank, :quality]
        ids = np.zeros(quality, np.uint32)
        for level, (feature, border, kind) in enumerate(zip(tree["features"], tree["borders"], tree["types"])):
            bins = data["banks"][bank, feature, order]
            ids |= np.asarray(bins == border if kind else bins > border, np.uint32) << level
        base, values, weights = cursors[offset:offset + quality], np.zeros(leaves, np.float32), np.zeros(leaves)
        if task in positions:
            original = data["weights"][order[:prefix]].astype(float)
            weights = np.bincount(ids[:prefix], weights=original, minlength=leaves)
            for evaluation in range(p.leaf_iterations):
                point = np.asarray(base[:prefix] + values[ids[:prefix]], np.float32)
                gradient, curvature = block_terms(data, order[:prefix], point, seeds[evaluation, positions[task]], options).T
                g = np.bincount(ids[:prefix], weights=gradient.astype(float), minlength=leaves)
                h = np.bincount(ids[:prefix], weights=curvature.astype(float), minlength=leaves)
                diagonal = h + p.l2 * (original.sum() if p.normalize else 1.)
                values = np.asarray(values + g / (diagonal + 1e-20), np.float32)
                values[weights < 1e-20] = 0.
            values = np.asarray(values - np.float32(values.mean(dtype=float)), np.float32)
        all_values.append(values); all_weights.append(weights)
        updated[offset:offset + quality] = np.asarray(base + values[ids] * np.float32(p.learning_rate), np.float32)
    predictions = np.empty(p.rows, np.float32)
    _, quality, offset, bank = descriptors[-1].astype(int)
    predictions[data["permutations"][bank]] = updated[offset:offset + quality]
    return updated, predictions, np.asarray(all_values), np.asarray(all_weights)


def advance(runtime, selected, weak, leaves):
    assert runtime.seed_shape(selected) == (len(weak), leaves.size)
    runtime.seeds(weak); runtime.call("begin_tree", selected)
    split = runtime.grow()
    assert split.has_split and split.finished and split.depth == 1
    runtime.seeds(leaves.ravel(), leaf=True)
    return runtime.finish()


@pytest.mark.parametrize("selected", [0, 1])
@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize("leaf_iterations,legacy", [(1, False), (1, True), (3, False), (3, True)])
def test_separate_weak_oracles_and_all_prefix_leaf_walks(yeti_library, selected, score, leaf_iterations, legacy):
    data = query_problem()
    p = query_params(data, 17, score=score, normalize=bool(selected), leaf_iterations=leaf_iterations)
    expected = expected_descriptors(data, p)
    with YetiRuntime(yeti_library, p, data, legacy=legacy) as runtime:
        descriptors, cursors = runtime.state()
        np.testing.assert_array_equal(descriptors, expected)
        weak, seeds = packet(data, p, expected, selected)
        assert runtime.seed_shape(selected) == (len(weak), seeds.size)
        score_expected = score_oracle(data, p, expected, cursors, selected, weak, runtime.options)
        # Yeti GradientAt aliases NewtonAt; Cosine also uses generated curvature.
        wrong = score_oracle(data, p, expected, cursors, selected, weak, runtime.options, object_weights=True)
        assert abs(score_expected - wrong) > 1e-4
        runtime.seeds(weak); runtime.call("begin_tree", selected)
        split = runtime.grow()
        assert split.has_split and split.depth == 1 and split.feature == split.bin == 0
        np.testing.assert_allclose(split.score, score_expected, rtol=8e-5, atol=2e-6)
        runtime.seeds(seeds.ravel(), leaf=True)
        tree = runtime.finish()
        updated, predictions, values, weights = leaf_oracle(data, p, expected, cursors, tree, seeds, runtime.options)
        np.testing.assert_allclose(runtime.state()[1], updated, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(runtime.predictions(), predictions, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(tree["values"], values[-1] * p.learning_rate, rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(tree["weights"], weights[-1], rtol=3e-6, atol=2e-6)
        assert tree["loss"] == 0


@pytest.mark.parametrize("leaf_iterations", [1, 3])
def test_explicit_packet_and_every_cursor_restore_exactly(yeti_library, leaf_iterations):
    data = query_problem(); p = query_params(data, 17, leaf_iterations=leaf_iterations)
    descriptors = expected_descriptors(data, p)
    with YetiRuntime(yeti_library, p, data, legacy=True) as runtime:
        advance(runtime, 1, *packet(data, p, descriptors, 1))
        saved = runtime.state()[1]
        expected = advance(runtime, 0, *packet(data, p, descriptors, 0, 1))
        final, predictions = runtime.state()[1], runtime.predictions()
    with YetiRuntime(yeti_library, p, data, legacy=True) as restored:
        restored.call("restore_cursors", len(saved), _f32(saved))
        actual = advance(restored, 0, *packet(data, p, descriptors, 0, 1))
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(restored.state()[1], final)
        np.testing.assert_array_equal(restored.predictions(), predictions)


def test_final_evaluation_consumes_a_packet_row_without_changing_values(yeti_library):
    data = query_problem(); p = query_params(data, 17, leaf_iterations=3)
    descriptors = expected_descriptors(data, p)
    weak, leaves = packet(data, p, descriptors, 1)
    changed = leaves.copy(); changed[-1] ^= np.uint64(0xfedcba9876543210)
    with YetiRuntime(yeti_library, p, data) as first, YetiRuntime(yeti_library, p, data) as second:
        expected, actual = advance(first, 1, weak, leaves), advance(second, 1, weak, changed)
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
        np.testing.assert_array_equal(first.state()[1], second.state()[1])


def singleton_prefix_problem():
    data = query_problem()
    sizes = np.asarray([1, 1, 3, 4, 5, 2, 6], np.uint32)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    orders = [[0, 1, 2, 3, 4, 5, 6], [1, 0, 5, 2, 3, 4, 6], [6, 4, 3, 2, 5, 1, 0]]
    permutations = np.asarray([[row for group in order for row in range(offsets[group], offsets[group + 1])]
                               for order in orders], np.uint32)
    rows = int(offsets[-1])
    data.update(sizes=sizes, offsets=offsets, group_orders=orders, permutations=permutations,
                banks=np.ascontiguousarray(data["banks"][:, :, :rows]), targets=data["targets"][:rows].copy(),
                weights=data["weights"][:rows].copy(), initial=data["initial"][:rows].copy())
    return data


def test_singleton_only_prefixes_skip_leaf_draws_and_updates(yeti_library):
    data = singleton_prefix_problem(); p = query_params(data, 17, leaf_iterations=3)
    descriptors = expected_descriptors(data, p)
    eligible = active_tasks(data, descriptors)
    assert len(eligible) < len(descriptors) and len(descriptors) - 1 in eligible
    with YetiRuntime(yeti_library, p, data) as runtime:
        np.testing.assert_array_equal(runtime.state()[0], descriptors)
        cursors = runtime.state()[1]
        weak, seeds = packet(data, p, descriptors, 0)
        assert runtime.seed_shape(0) == (len(weak), seeds.size)
        tree = advance(runtime, 0, weak, seeds)
        updated, _, values, _ = leaf_oracle(data, p, descriptors, cursors, tree, seeds, runtime.options)
        np.testing.assert_allclose(runtime.state()[1], updated, rtol=2e-4, atol=2e-6)
        for task, (_, quality, offset, _) in enumerate(descriptors.astype(int)):
            if task not in eligible:
                np.testing.assert_array_equal(values[task], 0.)
                np.testing.assert_array_equal(runtime.state()[1][offset:offset + quality], cursors[offset:offset + quality])


def test_empty_quality_keeps_its_seed_slot(yeti_library):
    data = query_problem()
    sizes = np.asarray([1, 1, 1, 497], np.uint32)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    orders = [[0, 1, 2, 3], [3, 0, 1, 2], [1, 3, 0, 2]]
    permutations = np.asarray([[row for group in order for row in range(offsets[group], offsets[group + 1])]
                               for order in orders], np.uint32)
    row = np.arange(int(offsets[-1]))
    data.update(sizes=sizes, offsets=offsets, group_orders=orders, permutations=permutations,
                banks=np.asarray([[(row + bank) % 3 == 0] for bank in range(3)], np.uint8),
                targets=np.asarray(((row * 7) % 13) / 13., np.float32),
                weights=np.ones(len(row), np.float32), initial=np.asarray(.17 * np.sin(row), np.float32))
    p = query_params(data, 17, leaf_iterations=1)
    descriptors, offset = [], 0
    for bank in range(p.permutations - 1):
        for prefix, quality in reference_group_folds(sizes[orders[bank]], data["growth"], p.min_fold_size):
            descriptors.append([prefix, quality, offset, bank]); offset += quality
    descriptors.append([p.rows, p.rows, offset, p.permutations - 1])
    descriptors = np.asarray(descriptors, np.uint32)
    assert tuple(descriptors[0, :2]) == (p.rows, p.rows)
    weak, leaves = packet(data, p, descriptors, 0)
    assert len(weak) == 2
    changed = weak.copy(); changed[1] ^= np.uint64(0xfedcba9876543210)
    results = []
    for seeds in (weak, changed):
        with YetiRuntime(yeti_library, p, data) as runtime:
            np.testing.assert_array_equal(runtime.state()[0], descriptors)
            assert runtime.seed_shape(0) == (2, leaves.size)
            runtime.seeds(seeds); runtime.call("begin_tree", 0)
            structure = runtime.grow()
            assert structure.finished and not structure.has_split
            runtime.seeds(leaves.ravel(), leaf=True)
            tree = runtime.finish()
            assert tree["depth"] == 0
            np.testing.assert_array_equal(tree["values"], 0.)
            np.testing.assert_array_equal(runtime.predictions(), data["initial"])
            results.append(runtime.state()[1])
    np.testing.assert_array_equal(*results)


@pytest.mark.parametrize("method,backtracking", [(1, None), (2, None), (0, 1), (0, 2)])
def test_yeti_requires_newton_and_no_backtracking(yeti_library, method, backtracking):
    data = query_problem(); p = query_params(data, 17, method=method)
    with pytest.raises(RuntimeError, match="(?i)(Newton|leaf|backtracking|Exact)"):
        with YetiRuntime(yeti_library, p, data) as runtime:
            if backtracking is None:
                pytest.fail("Ordered YetiRank accepted non-Newton leaves")
            runtime.call("set_backtracking", backtracking)


@pytest.mark.parametrize("kind", ["weak_short", "weak_long", "leaf_short", "leaf_long"])
def test_wrong_seed_packet_sizes_are_rejected(yeti_library, kind):
    data = query_problem(); p = query_params(data, 17)
    descriptors = expected_descriptors(data, p)
    weak, leaves = packet(data, p, descriptors, 0)
    with YetiRuntime(yeti_library, p, data) as runtime:
        source = leaves.ravel() if kind.startswith("leaf") else weak
        wrong = source[:-1] if kind.endswith("short") else np.r_[source, np.uint64(19)]
        if kind.startswith("leaf"):
            runtime.seeds(weak); runtime.call("begin_tree", 0); runtime.grow()
        with pytest.raises(RuntimeError, match="(?i)(seed|packet)"):
            runtime.seeds(wrong, leaf=kind.startswith("leaf"))
            if kind.startswith("weak"):
                runtime.call("begin_tree", 0)


def test_seed_shape_and_packet_boundaries(yeti_library):
    data = query_problem(); p = query_params(data, 17)
    descriptors = expected_descriptors(data, p)
    weak, leaves = packet(data, p, descriptors, 0)
    with YetiRuntime(yeti_library, p, data) as runtime:
        with pytest.raises(RuntimeError, match="(?i)(permutation|shape)"):
            runtime.seed_shape(p.permutations - 1)
        with pytest.raises(RuntimeError, match="(?i)(seed|begin|finish)"):
            runtime.seeds(leaves.ravel(), leaf=True)
        runtime.seeds(weak); runtime.call("begin_tree", 0)
        with pytest.raises(RuntimeError, match="(?i)(seed|begin)"):
            runtime.seeds(weak)
        runtime.grow()
        with pytest.raises(RuntimeError, match="(?i)(seed|packet)"):
            runtime.finish()


def test_missing_weak_packet_is_rejected(yeti_library):
    data = query_problem(); p = query_params(data, 17)
    with YetiRuntime(yeti_library, p, data) as runtime:
        with pytest.raises(RuntimeError, match="(?i)(seed|packet)"):
            runtime.call("begin_tree", 0)
