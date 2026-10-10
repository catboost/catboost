"""PairLogitPairwise graph projection and coupled solve on actual Metal."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from test_leaf_matrix_kernels import regularize


class PairMatrixParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "pairs", "leaves", "leaf_method", "reserved0", "reserved1", "reserved2", "reserved3")]


@pytest.fixture(scope="module")
def pair_matrix_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("pairwise_matrix_probe.mm")
    root = source.parent.parent
    dependencies = [source, *(root / "native" / name for name in (
        "metal_pairwise_matrix_kernels.h", "metal_leaf_matrix_kernels.h", "metal_sort.h",
        "metal_sort.mm", "metal_sort_kernels.h"))]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in dependencies)).hexdigest()[:16]
    destination = root / ".build" / f"pairwise_matrix_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        str(root / "native/metal_sort.mm"), "-o", str(destination)],
                       check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_pairwise_matrix_probe.argtypes = [ct.POINTER(PairMatrixParams), ct.c_float, ct.c_float] + [ct.c_void_p] * 14 + [ct.c_uint32]
    library.cbm_pairwise_matrix_probe.restype = ct.c_int

    def run(point, winners, losers, weights, leaf_ids, *, leaves=None, method="Newton", l2=3, non_diag_l2=.2):
        point, weights = np.ascontiguousarray(point, np.float32), np.ascontiguousarray(weights, np.float32)
        winners, losers, leaf_ids = (np.ascontiguousarray(a, np.uint32) for a in (winners, losers, leaf_ids))
        assert winners.shape == losers.shape == weights.shape
        assert point.shape == leaf_ids.shape
        leaves = int(leaf_ids.max()) + 1 if leaves is None else leaves
        edges = np.zeros((len(winners), 4), np.float32)
        keys, indices = np.zeros(len(winners), np.uint32), np.zeros(len(winners), np.uint32)
        offsets = np.zeros(leaves * leaves + 1, np.uint32)
        gradient, hessian, direction = np.zeros(leaves, np.float32), np.zeros((leaves, leaves), np.float32), np.zeros(leaves, np.float32)
        p = PairMatrixParams(len(point), len(winners), leaves, method == "Gradient", 0, 0, 0, 0)
        status, error = ct.c_uint32(), ct.create_string_buffer(4096)
        arrays = point, winners, losers, weights, leaf_ids, edges, keys, indices, offsets, gradient, hessian, direction
        code = library.cbm_pairwise_matrix_probe(ct.byref(p), l2, non_diag_l2,
            *(a.ctypes.data for a in arrays), ct.byref(status), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return dict(edges=edges, keys=keys, indices=indices, offsets=offsets,
                    gradient=gradient, hessian=hessian, direction=direction, status=status.value)
    return run


def edge_reference(point, winners, losers, weights):
    point, weights = np.asarray(point, np.float32), np.asarray(weights, np.float32)
    difference = point[winners] - point[losers]
    exponential = np.exp(-np.abs(difference))
    probability = np.where(difference >= 0, np.float32(1) / (np.float32(1) + exponential),
                           exponential / (np.float32(1) + exponential))
    probability = np.clip(probability, np.float32(1e-7), np.float32(1 - 1e-7))
    g = weights * (np.float32(1) - probability)
    h = (weights * probability) * (np.float32(1) - probability)
    loss = weights.astype(float) * np.logaddexp(0, -difference.astype(float))
    return np.column_stack((g, h, weights, loss))


def reference(point, winners, losers, weights, ids, leaves, method, l2=3, non_diag_l2=.2):
    edge = edge_reference(point, winners, losers, weights)
    win, lose = ids[winners], ids[losers]
    mask = win != lose
    gradient, hessian = np.zeros(leaves), np.zeros((leaves, leaves))
    g, h = edge[:, 0], edge[:, 2 if method == "Gradient" else 1]
    np.add.at(gradient, win[mask], g[mask]); np.add.at(gradient, lose[mask], -g[mask])
    np.add.at(hessian, (win[mask], win[mask]), h[mask]); np.add.at(hessian, (lose[mask], lose[mask]), h[mask])
    np.add.at(hessian, (win[mask], lose[mask]), -h[mask]); np.add.at(hessian, (lose[mask], win[mask]), -h[mask])
    key = np.where(mask, win * leaves + lose, np.uint32(0xffffffff)).astype(np.uint32)
    indices = np.argsort(key, kind="stable").astype(np.uint32)
    offsets = np.searchsorted(key[indices], np.arange(leaves * leaves + 1), side="left").astype(np.uint32)
    matrix = regularize(hessian, False, l2, non_diag_l2)
    direction = np.zeros(leaves)
    if leaves > 1:
        direction[:-1] = np.linalg.solve(matrix[:-1, :-1], gradient.astype(np.float32)[:-1])
    return dict(edges=edge, keys=key[indices], indices=indices, offsets=offsets,
                gradient=gradient, hessian=hessian, direction=direction)


def fixture(leaves, pairs=4099, rows=521, seed=739):
    rng = np.random.default_rng(seed)
    point = rng.normal(0, .8, rows).astype(np.float32)
    winners = rng.integers(0, rows, pairs, dtype=np.uint32)
    losers = (winners + rng.integers(1, rows, pairs, dtype=np.uint32)) % rows
    weights = rng.lognormal(0, .5, pairs).astype(np.float32)
    weights[::17] = 0
    ids = np.arange(rows, dtype=np.uint32) % leaves
    return point, winners, losers, weights, ids


@pytest.mark.parametrize("leaves", [1, 2, 7, 16, 64, 256])
@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_sorted_graph_projection_and_solve_match_independent_matrix(pair_matrix_probe, leaves, method):
    args = fixture(leaves)
    result, expected = pair_matrix_probe(*args, leaves=leaves, method=method), reference(*args, leaves, method)
    assert result["status"] == 0
    for key in ("keys", "indices", "offsets"):
        np.testing.assert_array_equal(result[key], expected[key])
    for key in ("edges", "gradient", "hessian", "direction"):
        np.testing.assert_allclose(result[key], expected[key], rtol=2e-5, atol=2e-4, err_msg=key)
    np.testing.assert_array_equal(result["hessian"], result["hessian"].T)
    np.testing.assert_allclose(result["hessian"].sum(axis=1, dtype=float), 0, atol=3e-4)
    assert result["direction"][-1] == 0


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_same_leaf_edges_are_filtered_before_full_laplacian(pair_matrix_probe, method):
    point = np.array([1, -1, 2, 0], np.float32)
    winners, losers = np.array([0, 1, 0]), np.array([1, 2, 2])
    weights, ids = np.array([1, 2, 3], np.float32), np.zeros(4, np.uint32)
    result = pair_matrix_probe(point, winners, losers, weights, ids, leaves=4, method=method)
    assert result["status"] == 0
    for key in ("gradient", "hessian", "direction", "offsets"):
        np.testing.assert_array_equal(result[key], 0)
    np.testing.assert_array_equal(result["keys"], np.uint32(0xffffffff))
    assert result["edges"][:, 1].sum() > 0  # Edge diagonal alone would be the wrong projected Hessian.


def test_clipped_pairwise_probability_retains_nonzero_extreme_curvature(pair_matrix_probe):
    point = np.array([-1e4, -100, 0, 100, 1e4, 0], np.float32)
    winners, losers = np.arange(5, dtype=np.uint32), np.full(5, 5, np.uint32)
    weights, ids = np.arange(1, 6, dtype=np.float32), np.arange(6, dtype=np.uint32)
    result = pair_matrix_probe(point, winners, losers, weights, ids)
    expected = edge_reference(point, winners, losers, weights)
    assert result["status"] == 0
    np.testing.assert_allclose(result["edges"], expected, rtol=3e-6, atol=1e-6)
    assert (result["edges"][:, :2] > 0).all()


@pytest.mark.parametrize("method", ["Newton", "Gradient"])
def test_high_degree_repeated_edges_and_reverse_order_remain_independent(pair_matrix_probe, method):
    args = fixture(7, pairs=65539, rows=31)
    result = pair_matrix_probe(*args, method=method)
    expected = reference(*args, 7, method)
    assert result["status"] == 0
    for key in ("gradient", "hessian", "direction"):
        np.testing.assert_allclose(result[key], expected[key], rtol=3e-5, atol=.003)
    order = np.arange(len(args[1]) - 1, -1, -1)
    reversed_args = args[0], *(a[order] for a in args[1:4]), args[4]
    changed = pair_matrix_probe(*reversed_args, method=method)
    for key in ("gradient", "hessian", "direction"):
        np.testing.assert_array_equal(result[key], changed[key])


def test_cancelling_projected_gradient_retains_small_remainder(pair_matrix_probe):
    result = pair_matrix_probe(np.zeros(3), [0, 0, 2], [1, 2, 0], [2**25, 2, 2**25], np.arange(3))
    assert result["status"] == 0
    assert result["gradient"][0] == 1


def test_empty_pair_projection_is_zero_and_deterministic(pair_matrix_probe):
    result = pair_matrix_probe(np.zeros(3), [], [], [], np.arange(3))
    assert result["status"] == 0
    for key in ("gradient", "hessian", "direction", "offsets"):
        np.testing.assert_array_equal(result[key], 0)


@pytest.mark.parametrize("bad", ["endpoint", "self_pair", "leaf", "weight", "point"])
def test_gpu_routing_and_arithmetic_status_prevent_invalid_projection(pair_matrix_probe, bad):
    point, winners, losers, weights, ids = fixture(4, pairs=7, rows=8)
    if bad == "endpoint": winners[0] = 123
    if bad == "self_pair": winners[0] = losers[0]
    if bad == "leaf": ids[0] = 17; winners[0] = 0; losers[0] = 1
    if bad == "weight": weights[0] = -1
    if bad == "point": point[0] = np.inf; winners[1] = 0; losers[1] = 1; weights[1] = 1
    result = pair_matrix_probe(point, winners, losers, weights, ids, leaves=4)
    assert result["status"] != 0
