"""Ordered stochastic backtracking uses an unbounded on-demand seed stream.

Missing-callback validation must precede leaf work and leave the pending tree
usable. One-iteration estimation still supports the fixed seed packet ABI.
"""
import ctypes as ct

import numpy as np
import pytest

from catboost_metal import _ordered
from catboost_metal._native import _f32, _u8, _u32
from test_combination_runtime import Component, Options, component
from test_ordered_query_runtime import expected_descriptors, query_library, query_params, query_problem
from test_ordered_training import apple_silicon, prohibit_cpu_training
from test_ordered_yeti_runtime import YetiRuntime, local_offsets, yeti_library
from test_yeti_rank_kernels import reference


SEED_CALLBACK = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ct.c_uint64))


@pytest.fixture(scope="module")
def combination_library(yeti_library):
    lib = yeti_library
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    signatures = {
        "create_combination_banked": [ct.POINTER(_ordered.Params), ct.c_uint32, ct.c_uint64, u8,
            f32, f32, f32, u32, u32, u8, u32, ct.POINTER(Options), ct.POINTER(Component),
            ct.c_uint32, u32, u32, u32, f32, ct.c_double, ct.POINTER(ct.c_void_p)],
        "set_combination_yeti_seed_callback": [ct.c_void_p, ct.c_void_p, ct.c_void_p],
    }
    for name, args in signatures.items():
        operation = getattr(lib, "cbm_ordered_session_" + name)
        operation.argtypes, operation.restype = args + [ct.c_char_p, ct.c_size_t], ct.c_int
    return lib


class CombinationRuntime(YetiRuntime):
    def __init__(self, lib, params, data, *, rmse_weight=100., yeti_weight=.2, one_hot=False):
        self.lib, self.params, self.data = lib, params, data
        self.handle = ct.c_void_p()
        options = Options(2, len(data["sizes"]), 0, 0)
        components = (Component * 2)(component("RMSE", rmse_weight),
            component("YetiRank", yeti_weight, permutations=7, decay=.71))
        features = np.arange(params.candidates, dtype=np.uint32)
        borders = np.full(params.candidates, int(one_hot), np.uint32)
        types = np.full(params.candidates, int(one_hot), np.uint8)
        error = ct.create_string_buffer(4096)
        code = lib.cbm_ordered_session_create_combination_banked(
            ct.byref(params), len(data["banks"]), data["banks"].size, _u8(data["banks"]),
            _f32(data["targets"]), _f32(data["weights"]), _f32(data["initial"]),
            _u32(features), _u32(borders), _u8(types), _u32(data["permutations"]),
            ct.byref(options), components, len(data["sizes"]), _u32(data["offsets"]),
            None, None, None, data["growth"], ct.byref(self.handle), error, len(error))
        if code:
            assert not self.handle.value
            raise RuntimeError(error.value.decode())
        assert self.handle.value


def leaf_seed(index):
    return 0x196A372AC010019 + 97 * index


def install_callback(runtime):
    consumed = []

    @SEED_CALLBACK
    def callback(context, output):
        seed = leaf_seed(len(consumed))
        consumed.append(seed)
        output[0] = seed
        return 0

    runtime.call("set_combination_yeti_seed_callback", callback, None)
    return callback, consumed


@pytest.mark.parametrize("leaf_iterations,backtracking,score", [(3, 1, 0), (3, 2, 1), (1, 2, 0)])
def test_callback_preflight_preserves_pending_tree_and_single_step_packet(
        combination_library, leaf_iterations, backtracking, score):
    data = query_problem()
    params = query_params(data, 19, leaf_iterations=leaf_iterations, depth=0, score=score)
    # CUDA combines signed row statistics before leaf projection. Zero object
    # weights can retain negative Yeti incidence while every slice sum is valid.
    seed, negative = 91323, False
    for prefix, end, _, bank in expected_descriptors(data, params)[:-1]:
        if bank != 0:
            continue
        for begin, finish in ((0, prefix), (prefix, end)):
            rows = data["permutations"][bank, begin:finish]
            if len(rows):
                hessian = reference(data["targets"][rows], data["weights"][rows], data["initial"][rows],
                    local_offsets(data, rows), permutations=7, decay=.71, seed=seed)[1][:, 1]
                weak = 100 * data["weights"][rows] - .2 * hessian
                assert np.all(np.isfinite(weak)) and weak.sum() > 0
                negative |= bool(np.any(weak < 0))
            seed += 1
    assert negative
    with CombinationRuntime(combination_library, params, data) as guarded, \
            CombinationRuntime(combination_library, params, data) as uninterrupted:
        for runtime in (guarded, uninterrupted):
            runtime.call("set_backtracking", backtracking)
            weak_count, leaf_count = runtime.seed_shape(0)
            runtime.seeds(np.arange(weak_count, dtype=np.uint64) + np.uint64(91323))
            runtime.call("begin_tree", 0)
        # A correctly sized fixed packet does not authorize stochastic trials.
        guarded.seeds([leaf_seed(index) for index in range(leaf_count)], leaf=True)
        if leaf_iterations > 1:
            with pytest.raises(RuntimeError, match="backtracking requires a seed callback"):
                guarded.finish()
            guarded_callback, guarded_consumed = install_callback(guarded)
        expected_callback, expected_consumed = install_callback(uninterrupted)
        actual, expected = guarded.finish(), uninterrupted.finish()
        for key in ("depth", "features", "borders", "types", "values", "weights", "loss"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        np.testing.assert_array_equal(guarded.predictions(), uninterrupted.predictions())
        for actual_state, expected_state in zip(guarded.state(), uninterrupted.state()):
            np.testing.assert_array_equal(actual_state, expected_state)
        task_count = leaf_count // (leaf_iterations + int(leaf_iterations > 1))
        assert len(expected_consumed) >= leaf_count
        assert len(expected_consumed) % task_count == 0
        if leaf_iterations > 1:
            assert guarded_consumed == expected_consumed
        else:
            assert len(expected_consumed) == leaf_count


def signed_split_scores(data, params, one_hot=False):
    """Pinned CUDA right-only clamp, plus two intentionally wrong variants."""
    numerators, norms, seed = np.zeros(3), np.full(3, 1e-20), 91323
    for prefix, end, _, bank in expected_descriptors(data, params)[:-1]:
        if bank != 0:
            continue
        slices = []
        for begin, finish in ((0, prefix), (prefix, end)):
            rows = data["permutations"][bank, begin:finish]
            yeti = reference(data["targets"][rows], data["weights"][rows], data["initial"][rows],
                local_offsets(data, rows), permutations=7, decay=.71, seed=seed)[1]
            gradients = 3 * data["weights"][rows] * (data["targets"][rows] - data["initial"][rows]) - 2 * yeti[:, 0]
            denominators = 3 * data["weights"][rows] - 2 * yeti[:, 1]
            slices.append((rows, gradients.astype(float), denominators.astype(float)))
            seed += 1
        for side in (False, True):
            statistics = []
            for rows, gradients, denominators in slices:
                mask = (data["banks"][0, 0, rows] > 0) == side
                statistics.append((gradients[mask].sum(), denominators[mask].sum()))
            for mode in range(3):  # 0=unclamped, 1=CUDA right-only, 2=both sides
                estimate_gradient, estimate_mass = statistics[0]
                quality_gradient, quality_mass = statistics[1]
                if mode == 2 or (mode == 1 and side != one_hot):
                    estimate_mass, quality_mass = max(estimate_mass, 0), max(quality_mass, 0)
                mu = estimate_gradient / (estimate_mass + params.l2) if estimate_mass > 0 else 0
                numerators[mode] += quality_gradient * mu
                norms[mode] += quality_mass * mu ** 2
    assert np.all(norms > 1e-15)
    return -numerators / np.sqrt(norms)


@pytest.mark.parametrize("score", [0, 1])
@pytest.mark.parametrize("negative_left", [False, True])
@pytest.mark.parametrize("one_hot", [False, True])
def test_signed_split_weights_clamp_only_the_cuda_complement(combination_library, score, negative_left, one_hot):
    data = query_problem()
    data["banks"][:] = 0
    data["banks"][:, :, [0, 8, 19]] = 1
    if negative_left:
        data["banks"] ^= 1
    params = query_params(data, 19, leaf_iterations=1, depth=1, score=score)
    unclamped, expected, both_clamped = signed_split_scores(data, params, one_hot)
    wrong = unclamped if negative_left == one_hot else both_clamped
    assert abs(expected - wrong) > .001, "the fixture must discriminate the signed aggregate rule"
    with CombinationRuntime(combination_library, params, data, rmse_weight=3, yeti_weight=2, one_hot=one_hot) as runtime:
        weak_count, _ = runtime.seed_shape(0)
        runtime.seeds(np.arange(weak_count, dtype=np.uint64) + np.uint64(91323))
        runtime.call("begin_tree", 0)
        split = runtime.grow()
        assert split.has_split and split.feature == 0 and split.bin == int(one_hot)
        assert split.type == int(one_hot)
        np.testing.assert_allclose(split.score, expected, rtol=8e-6, atol=1e-5)
        assert abs(split.score - wrong) > .001


@pytest.mark.parametrize("leaf_iterations,backtracking,normalize", [(1, 0, False), (3, 0, True), (3, 1, False), (3, 2, True)])
def test_finite_nonpositive_combination_diagonal_retains_zero_direction(
        combination_library, leaf_iterations, backtracking, normalize):
    data = query_problem()
    sizes = np.full(8, 2, np.uint32)
    group_orders = np.asarray([np.arange(8), np.arange(7, -1, -1), np.roll(np.arange(8), 3)], np.uint32)
    data.update(sizes=sizes, offsets=np.arange(0, 17, 2, dtype=np.uint32), group_orders=group_orders,
                targets=np.tile([0., 1.], 8).astype(np.float32), weights=np.ones(16, np.float32),
                initial=np.zeros(16, np.float32), banks=np.zeros((3, 1, 16), np.uint8),
                permutations=np.asarray([[row for group in groups for row in (2 * group, 2 * group + 1)]
                                         for groups in group_orders], np.uint32))
    params = query_params(data, 19, leaf_iterations=leaf_iterations, depth=0, normalize=normalize)
    params.l2 = .2 if normalize else 2.
    # Each two-row query has Yeti incident curvature .15 per row at zero,
    # independent of its generated permutation. The aggregate full Hessian is
    # 16*(.25 - 10*.15)=-20, while the outer row mass remains positive16.
    hessian = reference(data["targets"], data["weights"], data["initial"], data["offsets"],
                        permutations=7, decay=.71, seed=91323)[1][:, 1]
    np.testing.assert_allclose(hessian, .15, rtol=1e-6, atol=1e-7)
    diagonal = float((.25 - 10 * hessian).sum()) / (16 if normalize else 1) + params.l2
    assert diagonal < 0
    with CombinationRuntime(combination_library, params, data, rmse_weight=.25, yeti_weight=10) as runtime:
        runtime.call("set_backtracking", backtracking)
        weak_count, leaf_count = runtime.seed_shape(0)
        runtime.seeds(np.arange(weak_count, dtype=np.uint64) + np.uint64(91323))
        runtime.call("begin_tree", 0)
        callback, consumed = install_callback(runtime)
        result = runtime.finish()
        np.testing.assert_array_equal(result["values"], [0.])
        np.testing.assert_array_equal(result["weights"], [16.])
        np.testing.assert_array_equal(runtime.predictions(), np.zeros(16, np.float32))
        np.testing.assert_array_equal(runtime.state()[1], 0.)
        assert len(consumed) == leaf_count
