"""Given-edge CUDA PairLogit equations executed on actual Metal hardware."""

import ctypes as ct
import hashlib
import itertools
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class PairwiseParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "pairs", "objective", "apply_leaf_values", "leaves", "reserved0", "reserved1", "reserved2")]


@pytest.fixture(scope="module")
def pair_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("pairwise_probe.mm")
    root = source.parent.parent
    header = root / "native/metal_pairwise_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:16]
    destination = root / ".build" / f"pairwise_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_pairwise_probe.argtypes = [ct.POINTER(PairwiseParams), ct.c_uint32] + [ct.c_void_p] * 13 + [ct.c_uint32]
    library.cbm_pairwise_probe.restype = ct.c_int

    def run(cursor, winners, losers, weights=None, *, leaf_ids=None, leaf_values=None, groups=7):
        cursor = np.ascontiguousarray(cursor, np.float32)
        winners, losers = (np.ascontiguousarray(value, np.uint32) for value in (winners, losers))
        weights = np.ascontiguousarray(np.ones(len(winners)) if weights is None else weights, np.float32)
        assert winners.shape == losers.shape == weights.shape
        apply = leaf_values is not None
        leaf_ids = np.ascontiguousarray(np.zeros(len(cursor)) if leaf_ids is None else leaf_ids, np.uint32)
        leaf_values = np.ascontiguousarray([0] if leaf_values is None else leaf_values, np.float32)
        point = np.zeros_like(cursor)
        edges = np.zeros((len(winners), 4), np.float32)
        gradient, curvature, incident = np.zeros_like(cursor), np.zeros_like(cursor), np.zeros_like(cursor)
        partials = np.zeros((groups, 2), np.float32)
        params = PairwiseParams(len(cursor), len(winners), 14, apply, len(leaf_values), 0, 0, 0)
        error = ct.create_string_buffer(4096)
        buffers = (cursor, winners, losers, weights, leaf_ids, leaf_values, point, edges, gradient, curvature, incident, partials)
        code = library.cbm_pairwise_probe(ct.byref(params), groups,
            *(value.ctypes.data for value in buffers), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return dict(point=point, edges=edges, gradients=gradient, curvature=curvature,
                    incident_weights=incident, objective=partials.sum(axis=0, dtype=np.float64) * len(winners))
    return run


def reference(cursor, winners, losers, weights):
    cursor, weights = (np.asarray(value, np.float32).astype(np.float64) for value in (cursor, weights))
    winners, losers = np.asarray(winners, np.uint32), np.asarray(losers, np.uint32)
    difference = cursor[winners] - cursor[losers]
    probability = np.exp(-np.logaddexp(0, -difference)).astype(np.float32).astype(np.float64)
    probability = np.maximum(probability, 1e-40)
    edge_g = weights * (1 - probability)
    edge_h = weights * probability * (1 - probability)
    loss = weights * np.logaddexp(0, -difference)
    gradient, curvature, incident = np.zeros_like(cursor), np.zeros_like(cursor), np.zeros_like(cursor)
    np.add.at(gradient, winners, edge_g); np.add.at(gradient, losers, -edge_g)
    np.add.at(curvature, winners, edge_h); np.add.at(curvature, losers, edge_h)
    np.add.at(incident, winners, weights); np.add.at(incident, losers, weights)
    return dict(edges=np.column_stack((edge_g, edge_h, loss, weights)), gradients=gradient,
                curvature=curvature, incident_weights=incident, objective=np.array([loss.sum(), weights.sum()]))


def fixture(rows=259, pairs=1031, seed=8172):
    rng = np.random.default_rng(seed)
    point = rng.normal(0, 2, rows).astype(np.float32)
    winners = rng.integers(0, rows, pairs, dtype=np.uint32)
    losers = (winners + rng.integers(1, rows, pairs, dtype=np.uint32)) % rows
    weights = rng.lognormal(0, 2, pairs).astype(np.float32)
    weights[::7] = 0
    return point, winners, losers, weights


@pytest.mark.parametrize("rows,pairs", [(2, 1), (17, 33), (259, 1031), (1031, 8197)])
def test_pairwise_gpu_matches_cuda_equations(pair_probe, rows, pairs):
    args = fixture(rows, pairs)
    result, expected = pair_probe(*args), reference(*args)
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=2e-5, atol=3e-4)
    assert abs(result["gradients"].sum(dtype=np.float64)) < 1e-3
    assert result["incident_weights"].sum(dtype=np.float64) == pytest.approx(2 * expected["objective"][1], rel=1e-6)


def test_duplicate_and_reversed_edges_are_independent(pair_probe):
    point = np.array([.4, -.2, 1, 9], np.float32)
    winners = np.array([0, 0, 1, 0, 2], np.uint32)
    losers = np.array([1, 1, 0, 1, 1], np.uint32)
    weights = np.array([1, 2, .5, 0, 3], np.float32)
    result, expected = pair_probe(point, winners, losers, weights), reference(point, winners, losers, weights)
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=2e-6, atol=1e-6)
    assert result["gradients"][3] == result["curvature"][3] == result["incident_weights"][3] == 0
    np.testing.assert_array_equal(result["edges"][3], 0)


@pytest.mark.parametrize("groups", [1, 7, 257])
def test_high_degree_repeated_edges_and_stable_reduction(pair_probe, groups):
    rows, pairs = 17, 65539
    point, _, _, weights = fixture(rows, pairs)
    winners = np.zeros(pairs, np.uint32)
    losers = 1 + np.arange(pairs, dtype=np.uint32) % (rows - 1)
    winners[::3], losers[::3] = losers[::3].copy(), winners[::3].copy()
    result = pair_probe(point, winners, losers, weights, groups=groups)
    expected = reference(point, winners, losers, weights)
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=3e-6, atol=.02)
    repeated = pair_probe(point, winners, losers, weights, groups=groups)
    for key in result:
        np.testing.assert_array_equal(result[key], repeated[key])


def test_pair_reordering_preserves_math(pair_probe):
    args = fixture(17, 8197)
    order = np.random.default_rng(8942).permutation(len(args[1]))
    reordered = (args[0], *(value[order] for value in args[1:]))
    original, moved = pair_probe(*args), pair_probe(*reordered)
    for key in ("gradients", "curvature", "incident_weights", "objective"):
        np.testing.assert_allclose(original[key], moved[key], rtol=3e-6, atol=.002)
    np.testing.assert_array_equal(original["edges"][order], moved["edges"])


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_cancelling_incident_gradients_preserve_tiny_remainder(pair_probe, order):
    winners = np.array([0, 0, 2], np.uint32)[list(order)]
    losers = np.array([1, 2, 0], np.uint32)[list(order)]
    weights = np.array([2**25, 2, 2**25], np.float32)[list(order)]
    result = pair_probe([0, 0, 0], winners, losers, weights)
    # At p=.5 the three signed contributions to row0 are +2^24,+1,-2^24.
    # A plain float32 tree reduction loses the one for some edge orders.
    assert result["gradients"][0] == 1


def test_extreme_logits_are_finite_and_preserve_float_saturation(pair_probe):
    differences = np.array([-1e4, -100, -80, -20, 0, 20, 80, 100, 1e4], np.float32)
    point = np.r_[differences, np.zeros(len(differences), np.float32)]
    winners = np.arange(len(differences), dtype=np.uint32)
    losers = winners + len(differences)
    weights = np.arange(1, len(differences) + 1, dtype=np.float32)
    result, expected = pair_probe(point, winners, losers, weights), reference(point, winners, losers, weights)
    for key, value in expected.items():
        assert np.isfinite(result[key]).all()
        np.testing.assert_allclose(result[key], value, rtol=3e-6, atol=1e-6)
    np.testing.assert_array_equal(result["edges"][differences >= 20, :2], 0)
    assert result["edges"][5, 2] > 0  # stable softplus retains the small positive loss


def test_complete_edge_trial_and_incident_leaf_weights(pair_probe):
    point, winners, losers, weights = fixture(259, 1031)
    leaf_ids = np.arange(len(point), dtype=np.uint32) % 7
    leaves = np.linspace(-2, 3, 7).astype(np.float32)
    result = pair_probe(point, winners, losers, weights, leaf_ids=leaf_ids, leaf_values=leaves)
    expected_point = (point + leaves[leaf_ids]).astype(np.float32)
    np.testing.assert_array_equal(result["point"], expected_point)
    expected = reference(expected_point, winners, losers, weights)
    for key, value in expected.items():
        np.testing.assert_allclose(result[key], value, rtol=2e-5, atol=.001)
    leaf_mass = np.bincount(leaf_ids, weights=result["incident_weights"])
    assert leaf_mass.sum() == pytest.approx(2 * weights.sum(dtype=np.float64), rel=1e-6)


def test_same_leaf_retains_pointwise_diagonal_not_pairwise_laplacian(pair_probe):
    result = pair_probe([0, 0], [0], [1], [4], leaf_ids=[0, 0], leaf_values=[3])
    assert result["gradients"].sum() == 0
    assert result["curvature"].sum() == 2
    # In the full pairwise Hessian [[1,-1],[-1,1]], projecting both rows
    # to one leaf gives zero. This is intentionally the pointwise path.
    laplacian = np.array([[1, -1], [-1, 1]])
    assert np.ones(2) @ laplacian @ np.ones(2) == 0


def test_component_shift_invariance(pair_probe):
    point = np.array([.5, -2, 1, 3, 4, -.5], np.float32)
    winners, losers = np.array([0, 2, 3, 5]), np.array([1, 0, 4, 3])
    original = pair_probe(point, winners, losers)
    point[:3] += 2**20; point[3:] -= 2**18
    shifted = pair_probe(point, winners, losers)
    for key in original.keys() - {"point"}:
        np.testing.assert_array_equal(shifted[key], original[key])


def test_gradient_matches_objective_finite_difference(pair_probe):
    point, winners, losers, weights = fixture(17, 33)
    point *= .2
    result = pair_probe(point, winners, losers, weights)
    for row in [0, 3, 9, 16]:
        left, right = point.copy(), point.copy()
        left[row] -= .001; right[row] += .001
        derivative = (reference(right, winners, losers, weights)["objective"][0]
                      - reference(left, winners, losers, weights)["objective"][0]) / float(right[row] - left[row])
        assert result["gradients"][row] == pytest.approx(-derivative, rel=2e-5, abs=2e-5)


@pytest.mark.parametrize("pairs", [0, 17])
def test_empty_or_zero_weight_edges_produce_zero_targets(pair_probe, pairs):
    point, winners, losers, weights = fixture(17, pairs)
    weights[:] = 0
    result = pair_probe(point, winners, losers, weights)
    for key in result.keys() - {"point"}:
        np.testing.assert_array_equal(result[key], 0)


@pytest.mark.parametrize("winners,losers,weights", [([0], [0], [1]), ([3], [0], [1]),
    ([0], [1], [-1]), ([0], [1], [np.nan]), ([0], [1], [np.inf])])
def test_invalid_edges_are_rejected(pair_probe, winners, losers, weights):
    with pytest.raises(ValueError, match="Invalid pair"):
        pair_probe([0, 0, 0], winners, losers, weights)
