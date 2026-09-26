"""Reusable PairLogit helper transactions executed on the real Metal device."""

import ctypes as ct
import hashlib
import itertools
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from test_pairwise_kernels import fixture, reference


@pytest.fixture(scope="module")
def runtime_factory():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("pairwise_runtime_probe.mm")
    root = source.parent.parent
    files = [source, root / "native/metal_pairwise_runtime.h", root / "native/metal_pairwise_kernels.h"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()[:16]
    destination = root / ".build" / f"pairwise_runtime_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                   "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    library = ct.CDLL(str(destination))
    u32, ptr = ct.c_uint32, ct.c_void_p
    library.cbm_pairwise_runtime_create.argtypes = [u32, u32, ptr, ptr, ptr, u32, u32, u32] + [ptr] * 5 + [u32]
    library.cbm_pairwise_runtime_create.restype = ptr
    library.cbm_pairwise_runtime_evaluate.argtypes = [ptr] * 4 + [u32] * 4 + [ptr] * 7 + [u32]
    library.cbm_pairwise_runtime_evaluate.restype = ct.c_int
    library.cbm_pairwise_runtime_center.argtypes = [ptr, ptr, u32, u32] + [ptr] * 4 + [u32]
    library.cbm_pairwise_runtime_center.restype = ct.c_int
    library.cbm_pairwise_runtime_destroy.argtypes = [ptr]
    library.cbm_pairwise_runtime_destroy.restype = None
    instances = []

    class Runtime:
        def __init__(self, rows, winners, losers, weights=None, *, max_leaves=17, loss_groups=7,
                     group_offsets=None, group_count=None):
            self.rows, self.max_leaves = rows, max_leaves
            winners, losers = (np.ascontiguousarray(value, np.uint32) for value in (winners, losers))
            weights = np.ascontiguousarray(np.ones(len(winners)) if weights is None else weights, np.float32)
            assert winners.shape == losers.shape == weights.shape
            offsets = None if group_offsets is None else np.ascontiguousarray(group_offsets, np.uint32)
            count = (0 if offsets is None else len(offsets) - 1) if group_count is None else group_count
            assert offsets is None or len(offsets) >= count + 1
            self.incident_weights = np.zeros(rows, np.float32)
            total, allocated = ct.c_double(), ct.c_uint64()
            error = ct.create_string_buffer(4096)
            self.handle = library.cbm_pairwise_runtime_create(rows, len(winners), winners.ctypes.data,
                losers.ctypes.data, weights.ctypes.data, max_leaves, loss_groups, count,
                None if offsets is None else offsets.ctypes.data, self.incident_weights.ctypes.data,
                ct.byref(total), ct.byref(allocated), error, len(error))
            if not self.handle:
                raise ValueError(error.value.decode())
            self.total_incident_weight, self.allocated_bytes = total.value, allocated.value
            instances.append(self)

        def evaluate(self, cursor, *, leaf_values=None, leaf_ids=None, clear_status=True, dispatch_start=0,
                     allow_nonfinite_trial=False):
            cursor = np.ascontiguousarray(cursor, np.float32)
            apply_shift = leaf_values is not None
            leaves = np.ascontiguousarray([0] if leaf_values is None else leaf_values, np.float32)
            ids = np.ascontiguousarray(np.zeros(self.rows) if leaf_ids is None else leaf_ids, np.uint32)
            assert cursor.shape == ids.shape == (self.rows,)
            gradient, hessian, incident = (np.zeros(self.rows, np.float32) for _ in range(3))
            objective = np.zeros(2, np.float64)
            dispatches, allocated = ct.c_uint64(dispatch_start), ct.c_uint64()
            error = ct.create_string_buffer(4096)
            code = library.cbm_pairwise_runtime_evaluate(self.handle, cursor.ctypes.data,
                leaves.ctypes.data, ids.ctypes.data, len(leaves), apply_shift, clear_status, allow_nonfinite_trial,
                gradient.ctypes.data, hessian.ctypes.data, incident.ctypes.data, objective.ctypes.data,
                ct.byref(dispatches), ct.byref(allocated), error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return dict(gradients=gradient, curvature=hessian, incident_weights=incident,
                        objective=objective, dispatches=dispatches.value, allocated_bytes=allocated.value)

        def center(self, leaf_values, *, clear_status=True, dispatch_start=0):
            leaves = np.ascontiguousarray(leaf_values, np.float32)
            output = np.zeros_like(leaves)
            dispatches, allocated = ct.c_uint64(dispatch_start), ct.c_uint64()
            error = ct.create_string_buffer(4096)
            code = library.cbm_pairwise_runtime_center(self.handle, leaves.ctypes.data, len(leaves),
                clear_status, output.ctypes.data, ct.byref(dispatches), ct.byref(allocated), error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return output, dispatches.value, allocated.value

        def close(self):
            if self.handle:
                library.cbm_pairwise_runtime_destroy(self.handle)
                self.handle = None

    yield Runtime
    for instance in instances:
        instance.close()


def check_result(result, expected, *, atol=3e-4):
    for key in ("gradients", "curvature", "incident_weights", "objective"):
        np.testing.assert_allclose(result[key], expected[key], rtol=3e-6, atol=atol)


@pytest.mark.parametrize("rows,pairs", [(2, 1), (17, 33), (259, 1031), (1031, 8197)])
def test_given_edge_runtime_matches_independent_equations(runtime_factory, rows, pairs):
    point, winners, losers, weights = fixture(rows, pairs)
    # The tiny fixture deliberately has weight zero at edge zero.
    weights[0] = 1
    runtime = runtime_factory(rows, winners, losers, weights)
    result = runtime.evaluate(point)
    expected = reference(point, winners, losers, weights)
    check_result(result, expected)
    np.testing.assert_allclose(runtime.incident_weights, expected["incident_weights"], rtol=1e-6)
    assert runtime.total_incident_weight == pytest.approx(2 * weights.sum(dtype=np.float64), rel=1e-6)
    assert result["allocated_bytes"] == runtime.allocated_bytes > 0
    assert result["dispatches"] == 5  # Four derivative/validation launches and one loss reduction.


def test_supplied_pairs_keep_literal_weights_duplicate_order_and_isolated_rows(runtime_factory):
    point = np.array([.4, -.2, 1, 9, -1, 2], np.float32)
    winners, losers = [0, 0, 1, 0, 4], [1, 1, 0, 2, 5]
    weights = np.array([1, 2, .5, 0, 3], np.float32)
    runtime = runtime_factory(6, winners, losers, weights, group_offsets=[0, 4, 6])
    result = runtime.evaluate(point)
    check_result(result, reference(point, winners, losers, weights), atol=1e-6)
    np.testing.assert_array_equal(runtime.incident_weights, [3.5, 3.5, 0, 0, 3, 3])
    np.testing.assert_array_equal(result["incident_weights"], runtime.incident_weights)
    assert result["objective"][1] == pytest.approx(6.5)
    assert runtime.total_incident_weight == 13


@pytest.mark.parametrize("loss_groups", [1, 7, 257])
def test_high_degree_csr_and_loss_dispatch_layout(runtime_factory, loss_groups):
    rows, pairs = 17, 65539
    point, _, _, weights = fixture(rows, pairs)
    winners = np.zeros(pairs, np.uint32)
    losers = 1 + np.arange(pairs, dtype=np.uint32) % (rows - 1)
    winners[::3], losers[::3] = losers[::3].copy(), winners[::3].copy()
    runtime = runtime_factory(rows, winners, losers, weights, loss_groups=loss_groups)
    result = runtime.evaluate(point)
    check_result(result, reference(point, winners, losers, weights), atol=.02)
    repeated = runtime.evaluate(point)
    for key in result:
        np.testing.assert_array_equal(result[key], repeated[key])


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_reusable_csr_preserves_cancelling_gradient_remainder(runtime_factory, order):
    winners = np.array([0, 0, 2], np.uint32)[list(order)]
    losers = np.array([1, 2, 0], np.uint32)[list(order)]
    weights = np.array([2**25, 2, 2**25], np.float32)[list(order)]
    runtime = runtime_factory(3, winners, losers, weights)
    assert runtime.evaluate([0, 0, 0])["gradients"][0] == 1


def test_trial_shift_pipeline_reuse_and_transaction_accounting(runtime_factory):
    point, winners, losers, weights = fixture(259, 1031)
    runtime = runtime_factory(len(point), winners, losers, weights, max_leaves=17)
    first = runtime.evaluate(point, dispatch_start=79)
    base_dispatches = first["dispatches"] - 79
    for iteration, leaves_count in enumerate([7, 2, 17, 7]):
        ids = np.arange(len(point), dtype=np.uint32) % leaves_count
        leaves = np.linspace(-2, 3, leaves_count).astype(np.float32) / (iteration + 1)
        shifted = (point + leaves[ids]).astype(np.float32)
        result = runtime.evaluate(point, leaf_values=leaves, leaf_ids=ids, dispatch_start=79)
        check_result(result, reference(shifted, winners, losers, weights), atol=.001)
        assert result["allocated_bytes"] == runtime.allocated_bytes
        assert result["dispatches"] - 79 == base_dispatches
    after = runtime.evaluate(point, dispatch_start=79)
    for key in first:
        np.testing.assert_array_equal(first[key], after[key])


def test_same_leaf_pair_keeps_pointwise_curvature(runtime_factory):
    runtime = runtime_factory(2, [0], [1], [4])
    result = runtime.evaluate([0, 0], leaf_values=[3], leaf_ids=[0, 0])
    np.testing.assert_array_equal(result["gradients"], [2, -2])
    np.testing.assert_array_equal(result["curvature"], [1, 1])
    np.testing.assert_array_equal(result["incident_weights"], [4, 4])
    assert result["objective"][0] == pytest.approx(4 * np.log(2), rel=1e-6)


@pytest.mark.parametrize("leaves", [[2], [1, 7, -2, 10], [2, 0, 0, 0, 0, 0, 0]])
def test_centering_is_unweighted_including_empty_leaves(runtime_factory, leaves):
    runtime = runtime_factory(3, [0, 0], [1, 2], [1, 100], max_leaves=17)
    before = runtime.evaluate([0, 0, 0])
    values = np.array(leaves, np.float32)
    expected = (values.astype(np.float64) - values.mean(dtype=np.float64)).astype(np.float32)
    centered, dispatches, allocated = runtime.center(values, dispatch_start=61)
    np.testing.assert_allclose(centered, expected, rtol=1e-6, atol=1e-7)
    assert dispatches == 64
    assert allocated == runtime.allocated_bytes
    after = runtime.evaluate([0, 0, 0])
    check_result(after, before)


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_centering_compensates_large_cancellation(runtime_factory, order):
    runtime = runtime_factory(2, [0], [1])
    values = np.array([2**24, 1, -2**24], np.float32)[list(order)]
    centered, _, _ = runtime.center(values)
    expected = (values.astype(np.float64) - 1 / 3).astype(np.float32)
    np.testing.assert_allclose(centered, expected, rtol=0, atol=6e-8)
    assert centered[np.flatnonzero(values == 1)[0]] == pytest.approx(2 / 3, abs=6e-8)


@pytest.mark.parametrize("leaves_count", [257, 1031, 65536])
def test_centering_reduces_multiple_tiles_without_losing_the_remainder(runtime_factory, leaves_count):
    runtime = runtime_factory(2, [0], [1], max_leaves=leaves_count)
    values = np.zeros(leaves_count, np.float32)
    values[0], values[256], values[-1] = 2**24, 1, -(2**24)
    if leaves_count == 257:
        # Keep the small remainder distinct from the final tile's negative term.
        values[1] = 1
    expected = (values.astype(np.float64) - values.mean(dtype=np.float64)).astype(np.float32)
    centered, dispatches, allocated = runtime.center(values)
    np.testing.assert_allclose(centered, expected, rtol=1e-6, atol=1e-8)
    assert dispatches == 3
    assert allocated == runtime.allocated_bytes


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_gpu_nonfinite_point_status_is_sticky_and_recoverable(runtime_factory, bad):
    runtime = runtime_factory(3, [0], [1])
    # Row 2 is isolated: validation must cover the complete point buffer.
    with pytest.raises(ValueError, match="GPU status"):
        runtime.evaluate([0, 0, bad])
    with pytest.raises(ValueError, match="GPU status"):
        runtime.evaluate([0, 0, 0], clear_status=False)
    result = runtime.evaluate([0, 0, 0], clear_status=True)
    check_result(result, reference([0, 0, 0], [0], [1], [1]))


@pytest.mark.parametrize("mode", ["nonfinite_shift", "overflow_shift", "bad_leaf_id", "nonfinite_center",
                                  "overflow_edge_difference", "overflow_edge_loss"])
def test_gpu_trial_and_center_status_recovery(runtime_factory, mode):
    edge_weight = 1e30 if mode == "overflow_edge_loss" else 1
    runtime = runtime_factory(3, [0], [1], [edge_weight])
    with pytest.raises(ValueError, match="GPU status"):
        if mode == "nonfinite_shift":
            runtime.evaluate([0, 0, 0], leaf_values=[np.nan], leaf_ids=[0, 0, 0])
        elif mode == "overflow_shift":
            runtime.evaluate([3e38, 0, 0], leaf_values=[3e38], leaf_ids=[0, 0, 0])
        elif mode == "bad_leaf_id":
            runtime.evaluate([0, 0, 0], leaf_values=[0], leaf_ids=[0, 0, 1])
        elif mode == "overflow_edge_difference":
            runtime.evaluate([-3e38, 3e38, 0])
        elif mode == "overflow_edge_loss":
            runtime.evaluate([-1e20, 1e20, 0])
        else:
            runtime.center([0, np.inf, 1])
    centered, _, _ = runtime.center([1, 2, 3], clear_status=True)
    np.testing.assert_array_equal(centered, [-1, 0, 1])
    check_result(runtime.evaluate([0, 0, 0]), reference([0, 0, 0], [0], [1], [edge_weight]))


@pytest.mark.parametrize("mode", ["nan_candidate", "overflow_candidate", "overflow_edge_loss"])
def test_nonfinite_backtracking_candidate_can_be_rejected_without_poisoning_status(runtime_factory, mode):
    edge_weight = 1e30 if mode == "overflow_edge_loss" else 1
    runtime = runtime_factory(3, [0], [1], [edge_weight])
    if mode == "nan_candidate":
        cursor, leaves, ids = [0, 0, 0], [np.nan], [0, 0, 0]
    elif mode == "overflow_candidate":
        cursor, leaves, ids = [-3e38, 0, 0], [-3e38], [0, 0, 0]
    else:
        cursor, leaves, ids = [0, 0, 0], [-1e20, 1e20], [0, 1, 0]
    result = runtime.evaluate(cursor, leaf_values=leaves, leaf_ids=ids, allow_nonfinite_trial=True)
    assert not np.isfinite(result["objective"][0])
    assert np.isfinite(result["objective"][1])
    assert result["objective"][1] == pytest.approx(edge_weight, rel=1e-6)
    # The owner can reject this trial and reuse the last accepted point without
    # clearing a hard-error flag. The helper does not accept the candidate itself.
    recovered = runtime.evaluate([0, 0, 0], clear_status=False)
    check_result(recovered, reference([0, 0, 0], [0], [1], [edge_weight]))
    with pytest.raises(ValueError, match="GPU status"):
        runtime.evaluate(cursor, leaf_values=leaves, leaf_ids=ids, allow_nonfinite_trial=False)


def test_trial_permission_does_not_suppress_invalid_leaf_ids(runtime_factory):
    runtime = runtime_factory(3, [0], [1])
    with pytest.raises(ValueError, match="GPU status"):
        runtime.evaluate([0, 0, 0], leaf_values=[np.nan], leaf_ids=[0, 0, 1],
                         allow_nonfinite_trial=True)
    with pytest.raises(ValueError, match="GPU status"):
        runtime.evaluate([0, 0, 0], clear_status=False)
    check_result(runtime.evaluate([0, 0, 0]), reference([0, 0, 0], [0], [1], [1]))


def test_nonfinite_trial_permission_still_requires_positive_gpu_edge_mass(runtime_factory):
    tiny = np.nextafter(np.float32(0), np.float32(1))
    runtime = runtime_factory(2, [0, 0], [1, 1], [tiny, tiny])
    # Mean-partial normalization underflows the smallest subnormal edge mass.
    # Permitting a nonfinite numerator must not permit a zero denominator.
    with pytest.raises(ValueError, match="zero edge mass"):
        runtime.evaluate([0, 0], allow_nonfinite_trial=True)


@pytest.mark.parametrize("winners,losers,weights", [([0], [0], [1]), ([3], [0], [1]),
    ([0], [3], [1]), ([0], [1], [-1]), ([0], [1], [np.nan]), ([0], [1], [np.inf]),
    ([0], [1], [0]), ([0], [1], [2e38]), ([], [], [])])
def test_constructor_rejects_invalid_edges_and_nonpositive_mass(runtime_factory, winners, losers, weights):
    with pytest.raises(ValueError):
        runtime_factory(3, winners, losers, weights)


@pytest.mark.parametrize("options", [dict(max_leaves=0), dict(max_leaves=65537),
    dict(loss_groups=0), dict(loss_groups=4097),
    dict(group_offsets=[1, 3]), dict(group_offsets=[0, 2]),
    dict(group_offsets=[0, 2, 1, 3]), dict(group_offsets=[0, 1, 1, 3]),
    dict(group_offsets=[0, 4, 3]), dict(group_count=1)])
def test_constructor_rejects_invalid_dimensions_and_group_offsets(runtime_factory, options):
    with pytest.raises(ValueError):
        runtime_factory(3, [0], [1], **options)


def test_cross_group_edges_and_zero_rows_are_rejected(runtime_factory):
    with pytest.raises(ValueError):
        runtime_factory(4, [0], [2], group_offsets=[0, 2, 4])
    with pytest.raises(ValueError):
        runtime_factory(0, [], [])


def test_allocated_bytes_account_for_reusable_graph_buffers(runtime_factory):
    small = runtime_factory(2, [0], [1], max_leaves=4, loss_groups=1)
    pairs = 8197
    large = runtime_factory(259, np.zeros(pairs, np.uint32),
                            1 + np.arange(pairs, dtype=np.uint32) % 258,
                            max_leaves=4, loss_groups=1)
    assert large.allocated_bytes > small.allocated_bytes + pairs * 4
    # Caller buffers and host metadata are deliberately outside helper accounting.
    assert small.allocated_bytes == 44 + 8 * 2 + 4 + 8 + 8 + 8
    assert large.allocated_bytes == 44 * pairs + 8 * 259 + 4 + 8 + 8 + 8
    original = large.allocated_bytes
    for _ in range(3):
        result = large.evaluate(np.zeros(259, np.float32))
        assert result["allocated_bytes"] == original
        assert result["objective"][1] == pytest.approx(pairs)
    _, _, allocated = large.center([1, 2, 3, 4])
    assert allocated == original
