"""Actual GPU sampled row counts, independent of original observation weights."""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("rows", "leaves", "bootstrap_type", "reserved")]


@pytest.fixture(scope="module")
def probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    source = Path(__file__).with_name("greedy_bootstrap_probe.mm")
    header = source.parent.parent / "native/metal_greedy_bootstrap_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:16]
    path = source.parent.parent / ".build" / f"greedy_bootstrap_probe_{digest}.dylib"
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        completed = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
    lib = ct.CDLL(str(path))
    lib.cbm_greedy_bootstrap_counts_probe.argtypes = [ct.POINTER(Params)] + [ct.c_void_p]*5 + [ct.c_uint32]
    lib.cbm_greedy_bootstrap_counts_probe.restype = ct.c_int
    lib.cbm_greedy_noise_statistics_probe.argtypes = [ct.POINTER(Params), ct.c_void_p,
        ct.c_void_p, ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.c_uint32]
    lib.cbm_greedy_noise_statistics_probe.restype = ct.c_int

    def invoke(multipliers, rows, offsets, bootstrap_type, *, params=None):
        multipliers = np.ascontiguousarray(multipliers, np.float32)
        rows, offsets = [np.ascontiguousarray(v, np.uint32) for v in (rows, offsets)]
        result = np.full(len(offsets), np.iinfo(np.uint32).max, np.uint32)
        p = Params(len(rows), len(offsets)-1, bootstrap_type, 0) if params is None else params
        error = ct.create_string_buffer(2048)
        code = lib.cbm_greedy_bootstrap_counts_probe(ct.byref(p), multipliers.ctypes.data, rows.ctypes.data,
            offsets.ctypes.data, result.ctypes.data, error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return result

    def noise(gradient, weight, groups, *, params=None):
        gradient, weight = [np.ascontiguousarray(v, np.float32) for v in (gradient, weight)]
        result = np.full((groups if 0 < groups <= 65536 else 1, 4), np.nan, np.float32)
        p = Params(len(gradient), 1, 0, 0) if params is None else params
        error = ct.create_string_buffer(2048)
        code = lib.cbm_greedy_noise_statistics_probe(ct.byref(p), gradient.ctypes.data, weight.ctypes.data,
            groups, result.ctypes.data, error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return result
    invoke.noise = noise
    return invoke


@pytest.mark.parametrize("bootstrap_type", range(4))
@pytest.mark.parametrize("sizes", [
    [1], [0, 3, 0, 4, 0], [0, 255, 1, 256, 257, 0], [0, 8193, 0, 257, 1, 0],
    [0, 1, 0, 3]*129, [0, 0, 0, 5] + [0]*65532,
])
def test_sampled_prefix_counts_arbitrary_partitions(probe, bootstrap_type, sizes):
    rng = np.random.default_rng(2801)
    rows = rng.permutation(sum(sizes)).astype(np.uint32)
    multipliers = rng.integers(0, 4, len(rows)).astype(np.float32)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    expected_counts = [size if bootstrap_type < 2 else np.count_nonzero(multipliers[rows[a:a+size]] > 0)
        for a, size in zip(offsets[:-1], sizes)]
    actual = probe(multipliers, rows, offsets, bootstrap_type)
    np.testing.assert_array_equal(actual, np.r_[0, np.cumsum(expected_counts)])


@pytest.mark.parametrize("bootstrap_type", [2, 3])
def test_sampled_zero_original_weights_still_count(probe, bootstrap_type):
    # Every retained draw below multiplies an original zero weight. Counting
    # positive structure weights would produce zero instead of [1,2,1].
    multipliers = np.array([1, 0, 3, 1, 0, 2], np.float32)
    original_weights = np.array([0, 2, 0, 0, 9, 0], np.float32)
    assert not np.any(multipliers * original_weights)
    result = probe(multipliers, np.arange(6), [0, 2, 4, 6], bootstrap_type)
    np.testing.assert_array_equal(result, [0, 1, 3, 4])


@pytest.mark.parametrize("bootstrap_type", range(4))
def test_all_zero_draw_has_no_retry_and_bayesian_retains_rows(probe, bootstrap_type):
    result = probe(np.zeros(8193), np.arange(8193)[::-1], [0, 0, 257, 8193, 8193], bootstrap_type)
    np.testing.assert_array_equal(result, [0, 0, 257, 8193, 8193] if bootstrap_type < 2 else np.zeros(5, int))


@pytest.mark.parametrize("params", [Params(0, 1, 2, 0), Params(2**24+1, 1, 2, 0),
    Params(2, 0, 2, 0), Params(2, 65537, 2, 0), Params(2, 1, 4, 0), Params(2, 1, 2, 1)])
def test_reject_invalid_dimensions_before_buffer_reads(probe, params):
    with pytest.raises(ValueError, match="Invalid"):
        probe([1, 0], [0, 1], [0, 2], 2, params=params)


@pytest.mark.parametrize("multipliers,rows,offsets", [
    ([1, -1], [0, 1], [0, 2]), ([1, float("nan")], [0, 1], [0, 2]),
    ([1, 0], [0, 0], [0, 2]), ([1, 0], [0, 2], [0, 2]),
    ([1, 0], [0, 1], [1, 2]), ([1, 0], [0, 1], [0, 1]),
    ([1, 0], [0, 1], [0, 2, 1, 2]),
])
def test_reject_invalid_partition_or_multiplier_inputs(probe, multipliers, rows, offsets):
    with pytest.raises(ValueError):
        probe(multipliers, rows, offsets, 2)


@pytest.mark.parametrize("rows", [1, 257, 8193, 16641])
@pytest.mark.parametrize("groups", [1, 3, 33])
def test_noise_uses_post_bootstrap_structure_weight_denominator(probe, rows, groups):
    rng = np.random.default_rng(145)
    original_weights = rng.uniform(.1, 7, rows).astype(np.float32)
    multipliers = rng.integers(0, 5, rows).astype(np.float32)
    weights = original_weights * multipliers
    gradients = rng.normal(size=rows).astype(np.float32) * weights
    keep = weights > np.float32(1e-15)
    # CUDA computes each g*g/w term in float32, then accumulates totals.
    terms = gradients[keep] * gradients[keep] / weights[keep]
    expected = [terms.sum(dtype=np.float64), weights[keep].sum(dtype=np.float64)]
    parts = probe.noise(gradients, weights, groups).astype(np.float64)
    actual = parts[:, :2].sum(axis=0) + parts[:, 2:].sum(axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)


def test_noise_strict_weight_threshold_and_zero_weight_gradients(probe):
    threshold = np.float32(1e-15)
    weight = np.array([0, threshold, np.nextafter(threshold, np.float32(0)),
        np.nextafter(threshold, np.float32(1)), 2], np.float32)
    gradient = np.array([1e15, 1e15, 1e15, 1e-5, 3], np.float32)
    parts = probe.noise(gradient, weight, 7).astype(np.float64)
    expected = [np.sum(gradient[3:]**2 / weight[3:], dtype=np.float64), np.sum(weight[3:], dtype=np.float64)]
    np.testing.assert_allclose(parts[:, :2].sum(0) + parts[:, 2:].sum(0), expected, rtol=1e-7)


@pytest.mark.parametrize("groups", [1, 3, 33])
def test_noise_all_filtered_draw_is_finite_zero(probe, groups):
    parts = probe.noise(np.full(8193, 1e18), np.zeros(8193), groups)
    np.testing.assert_array_equal(parts, np.zeros((groups, 4)))


def test_noise_preserves_high_low_expansions_across_reduction(probe):
    parts = probe.noise([4096, 1, 0], [1, 1, 2**24], 1).astype(np.float64)
    np.testing.assert_array_equal(parts[:, :2].sum(0) + parts[:, 2:].sum(0), [2**24+1, 2**24+2])
    assert parts[0, 2] != 0


@pytest.mark.parametrize("groups", [0, 65537, 2**32-1])
def test_noise_rejects_unsafe_group_dimensions(probe, groups):
    with pytest.raises(ValueError, match="group count"):
        probe.noise([1], [1], groups)


@pytest.mark.parametrize("gradient,weight", [([float("nan")], [1]), ([float("inf")], [1]),
    ([1], [float("nan")]), ([1], [-1])])
def test_noise_rejects_nonfinite_or_negative_inputs(probe, gradient, weight):
    with pytest.raises(ValueError):
        probe.noise(gradient, weight, 1)
