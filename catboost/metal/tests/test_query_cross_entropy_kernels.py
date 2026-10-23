"""CUDA QueryCrossEntropy statistics and full leaf Hessian on real Metal."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import expit


class Params(ct.Structure):
    _fields_ = [("rows", ct.c_uint32), ("groups", ct.c_uint32),
                ("leaves", ct.c_uint32), ("alpha", ct.c_float)]


@pytest.fixture(scope="module")
def probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("requires Apple Silicon Metal")
    root = Path(__file__).resolve().parents[1]
    source = root / "tests/query_cross_entropy_probe.mm"
    header = root / "native/metal_query_cross_entropy_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:20]
    library = root / ".build" / ("query_cross_entropy_" + digest + ".dylib")
    library.parent.mkdir(exist_ok=True)
    if not library.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(library)],
                       capture_output=True, text=True, check=True)
    lib = ct.CDLL(str(library))
    f32, u32 = ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint32)
    lib.cbm_qce_probe.argtypes = [ct.POINTER(Params), f32, f32, f32, u32, f32, u32,
                                f32, f32, u32, f32, f32, ct.c_char_p, ct.c_uint32]
    lib.cbm_qce_probe.restype = ct.c_int

    def run(y, w, x, offsets, leaves, *, alpha=.95, scales=None, leaf_count=None):
        y, w, x = [np.ascontiguousarray(v, np.float32) for v in (y, w, x)]
        offsets, leaves = [np.ascontiguousarray(v, np.uint32) for v in (offsets, leaves)]
        groups, count = len(offsets) - 1, int(np.max(leaves)) + 1 if leaf_count is None else leaf_count
        scales = np.ones(groups, np.float32) if scales is None else np.ascontiguousarray(scales, np.float32)
        assert y.shape == w.shape == x.shape == leaves.shape and scales.shape == (groups,)
        rows, queries = np.empty((len(y), 4), np.float32), np.empty((groups, 4), np.float32)
        single = np.empty(groups, np.uint32)
        gradient, hessian = np.empty(count, np.float32), np.empty((count, count), np.float32)
        error = ct.create_string_buffer(4096)
        p = Params(len(y), groups, count, alpha)
        code = lib.cbm_qce_probe(ct.byref(p), *(v.ctypes.data_as(f32) for v in (y, w, x)),
            offsets.ctypes.data_as(u32), scales.ctypes.data_as(f32), leaves.ctypes.data_as(u32),
            rows.ctypes.data_as(f32), queries.ctypes.data_as(f32), single.ctypes.data_as(u32),
            gradient.ctypes.data_as(f32), hessian.ctypes.data_as(f32), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return rows, queries, single, gradient, hessian
    return run


def reference(y, w, x, offsets, ids, *, alpha=.95, scales=None, leaf_count=None):
    """Independent constrained optimum plus explicit CUDA pair Laplacian."""
    y, w, x = [np.asarray(v, np.float32).astype(np.float64) for v in (y, w, x)]
    alpha = float(np.float32(alpha))
    count = int(max(ids)) + 1 if leaf_count is None else leaf_count
    rows, queries, singles = np.zeros((len(y), 4)), [], []
    matrix = np.zeros((count, count))
    for q, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        select = slice(start, end)
        target, weight, point = y[select], w[select], x[select]
        scale = 1 if scales is None else float(np.float32(scales[q]))
        single = np.all(np.abs(target - target[0]) <= np.float32(1e-5))
        shift = 0.
        if not single and weight.sum():
            def derivative(value):
                return np.sum(weight * (target - expit(point * scale + value)))
            shift = (-20 if derivative(-20) <= 0 else 20 if derivative(20) >= 0
                     else brentq(derivative, -20, 20, xtol=1e-13))
        shifted = point * scale + shift
        probability, shifted_probability = expit(point), expit(shifted)
        gradient = weight * ((1-alpha) * (target - probability)
                            + (0 if single else alpha * (target - shifted_probability) * scale))
        point_h = weight * (1-alpha) * probability * (1-probability)
        shifted_h = np.zeros(len(point)) if single else weight * alpha * shifted_probability * (1-shifted_probability) * scale**2
        loss = weight * ((1-alpha) * (np.logaddexp(0, point) - target * point)
                         + (0 if single else alpha * (np.logaddexp(0, shifted) - target * shifted)))
        rows[select] = np.column_stack((gradient, point_h, shifted_h, loss))
        queries.append([shift, shifted_h.sum(), loss.sum(), weight.sum()])
        singles.append(single)
        for i in range(start, end):
            a = ids[i]
            matrix[a, a] += point_h[i-start]
            for j in range(start, i):
                b = ids[j]
                if a == b or shifted_h.sum() <= 1e-20:
                    continue
                pair = shifted_h[i-start] * shifted_h[j-start] / (shifted_h.sum() + 1e-20)
                matrix[a, a] += pair
                matrix[b, b] += pair
                matrix[a, b] -= pair
                matrix[b, a] -= pair
    return rows, np.array(queries), np.array(singles, np.uint32), np.bincount(ids, weights=rows[:, 0], minlength=count), matrix


@pytest.mark.parametrize("sizes", [(1, 2, 3), (7, 32, 65), (255, 256)])
@pytest.mark.parametrize("alpha", [0, .37, .95, 1])
@pytest.mark.parametrize("soft", [False, True])
def test_weighted_statistics_and_full_laplacian(probe, sizes, alpha, soft):
    rng = np.random.default_rng(7992)
    offsets = np.r_[0, np.cumsum(sizes)]
    rows = offsets[-1]
    y = rng.uniform(.1, .9, rows) if soft else rng.integers(0, 2, rows)
    w, x = rng.uniform(.2, 2, rows), rng.normal(0, .8, rows)
    w[::19] = 0
    leaves = rng.integers(0, 5, rows)
    scales = rng.uniform(.5, 1.3, len(sizes))
    args = (y, w, x, offsets, leaves)
    options = dict(alpha=alpha, scales=scales, leaf_count=7)
    actual, expected = probe(*args, **options), reference(*args, **options)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-5)
    np.testing.assert_allclose(actual[4], actual[4].T, atol=1e-7)
    assert np.linalg.eigvalsh(actual[4]).min() > -1e-5
    np.testing.assert_array_equal(actual[4][5:], 0)


def test_same_leaf_cancels_query_hessian_and_gradient(probe):
    rows, groups, single, g, h = probe([0, 1, 0, 1], [1, 2, 3, 4], [-1, .4, .2, -.7], [0, 4], [0, 0, 0, 0], alpha=1)
    assert not single[0]
    assert abs(g[0]) < 2e-6
    assert h[0, 0] == 0
    assert groups[0, 1] > 0


def test_full_leaf_hessian_matches_optimized_loss_curvature(probe):
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    w = np.array([.3, 1.2, .8, 2, .7, 1.1, .4, 1.3])
    x = np.array([.2, -.3, .7, .1, -.6, .4, -.2, .8])
    offsets, ids = np.array([0, 4, 8]), np.array([0, 1, 2, 0, 2, 1, 0, 1])
    _, _, _, gradient, matrix = probe(y, w, x, offsets, ids, scales=[.8, 1.2])
    epsilon = .002
    finite_difference = np.empty((3, 3))
    for leaf in range(3):
        shift = epsilon * (ids == leaf)
        positive = reference(y, w, x + shift, offsets, ids, scales=[.8, 1.2])
        negative = reference(y, w, x - shift, offsets, ids, scales=[.8, 1.2])
        finite_difference[:, leaf] = -(positive[3] - negative[3]) / (2 * epsilon)
        assert -(positive[1][:, 2].sum() - negative[1][:, 2].sum()) / (2 * epsilon) == pytest.approx(
            gradient[leaf], abs=2e-5)
    np.testing.assert_allclose(matrix, finite_difference, rtol=1e-4, atol=2e-5)


def test_single_class_and_zero_weight_queries(probe):
    args = ([.3, .3, 0, 1, 0, 1], [1, 2, 0, 0, 0, 0], [-2, 3, 1, 2, 3, 4], [0, 2, 6], [0, 1, 0, 1, 0, 1])
    rows, groups, single, g, h = probe(*args)
    np.testing.assert_array_equal(single, [1, 0])
    np.testing.assert_array_equal(rows[:, 2], 0)
    np.testing.assert_array_equal(rows[2:], 0)
    assert np.isfinite(groups).all()
    np.testing.assert_allclose(h, np.diag(np.bincount(args[4], weights=rows[:, 1])), atol=1e-7)


def test_extreme_logits_remain_finite_and_probabilities_follow_cuda_clipping(probe):
    x = np.array([-10000, 10000, -100, 100], np.float32)
    y = np.array([1, 0, 0, 1], np.float32)
    rows, groups, single, g, h = probe(y, np.ones(4), x, [0, 4], np.arange(4), alpha=0)
    assert all(np.isfinite(v).all() for v in (rows, groups, g, h))
    probability = np.clip(expit(x), np.float32(1e-7), np.float32(1) - np.float32(1e-7))
    np.testing.assert_allclose(rows[:, 0], y - probability, rtol=1e-7, atol=1e-7)
    assert rows[:, 3].sum() == pytest.approx(20000, abs=1e-5)


@pytest.mark.parametrize("case", ["oversize", "negative_weight", "leaf", "target", "scale"])
def test_native_boundary_rejects_invalid_query_input(probe, case):
    size = 257 if case == "oversize" else 3
    y, w, x = np.zeros(size), np.ones(size), np.zeros(size)
    leaves = np.zeros(size, np.uint32)
    scales = [np.inf if case == "scale" else 1]
    if case == "negative_weight": w[0] = -1
    if case == "target": y[0] = 2
    if case == "leaf": leaves[0] = 2
    with pytest.raises(ValueError):
        probe(y, w, x, [0, size], leaves, scales=scales, leaf_count=2)
