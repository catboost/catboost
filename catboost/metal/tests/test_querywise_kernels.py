"""CUDA query objective algebra on real Metal; no CPU CatBoost training."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class QuerywiseParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("rows", "groups", "objective", "apply_leaf_values")] + [
        ("beta", ct.c_float), ("lambda_reg", ct.c_float), ("leaves", ct.c_uint32), ("reserved", ct.c_uint32)]


class QuerywiseProjectionParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("rows", "leaves", "tiles", "leaf_method")]


@pytest.fixture(scope="module")
def query_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("querywise_probe.mm")
    root = source.parent.parent
    headers = [root / "native" / name for name in ("metal_querywise_kernels.h", "metal_kernels.h",
        "metal_objective_kernels.h", "metal_additional_objective_kernels.h", "metal_kernel_abi.h")]
    digest = hashlib.sha256(source.read_bytes() + b"".join(header.read_bytes() for header in headers)).hexdigest()[:16]
    destination = root / ".build" / f"querywise_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_querywise_probe.argtypes = [ct.POINTER(QuerywiseParams)] + [ct.c_void_p] * 11 + [ct.c_uint32]
    library.cbm_querywise_probe.restype = ct.c_int
    library.cbm_querywise_project_probe.argtypes = [ct.POINTER(QuerywiseProjectionParams)] + [ct.c_void_p] * 8 + [
        ct.c_float, ct.c_void_p, ct.c_uint32]
    library.cbm_querywise_project_probe.restype = ct.c_int
    library.cbm_querywise_objective_probe.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32,
        ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_uint32]
    library.cbm_querywise_objective_probe.restype = ct.c_int
    library.cbm_querywise_curvature_probe.argtypes = [ct.c_void_p, ct.c_uint32,
        ct.c_void_p, ct.c_void_p, ct.c_uint32]
    library.cbm_querywise_curvature_probe.restype = ct.c_int

    def run(objective, targets, weights, cursor, offsets, *, beta=1, lambda_reg=.01,
            leaf_values=None, leaf_ids=None, reserved=0):
        targets, weights, cursor = (np.ascontiguousarray(value, np.float32) for value in (targets, weights, cursor))
        offsets = np.ascontiguousarray(offsets, np.uint32)
        assert targets.shape == weights.shape == cursor.shape
        apply = leaf_values is not None
        leaf_values = np.ascontiguousarray([0] if leaf_values is None else leaf_values, np.float32)
        leaf_ids = np.ascontiguousarray(np.zeros(len(cursor)) if leaf_ids is None else leaf_ids, np.uint32)
        params = QuerywiseParams(len(cursor), len(offsets) - 1, objective, apply, beta, lambda_reg, len(leaf_values), reserved)
        point = np.zeros_like(cursor)
        gradient, curvature = np.zeros_like(cursor), np.zeros_like(cursor)
        stats = np.zeros((len(offsets) - 1, 2), np.float32)
        error = ct.create_string_buffer(4096)
        arguments = [value.ctypes.data for value in (targets, weights, cursor, offsets, leaf_values,
                                                     leaf_ids, point, gradient, curvature, stats)]
        code = library.cbm_querywise_probe(ct.byref(params), *arguments, error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return point, gradient, curvature, stats
    def project(gradient, curvature, weights, row_indices, offsets, *, tiles=1, leaf_method=0, l2=2):
        gradient, curvature, weights = (np.ascontiguousarray(value, np.float32)
            for value in (gradient, curvature, weights))
        row_indices, offsets = (np.ascontiguousarray(value, np.uint32) for value in (row_indices, offsets))
        assert gradient.shape == curvature.shape == weights.shape == row_indices.shape
        leaves = len(offsets) - 1
        params = QuerywiseProjectionParams(len(gradient), leaves, tiles, leaf_method)
        partials = np.full((leaves, tiles, 2, 4), np.nan, np.float32)
        values, leaf_weights = np.zeros(leaves, np.float32), np.zeros(leaves, np.float32)
        error = ct.create_string_buffer(4096)
        args = [value.ctypes.data for value in (gradient, curvature, weights, row_indices, offsets,
            partials, values, leaf_weights)]
        code = library.cbm_querywise_project_probe(ct.byref(params), *args, l2, error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return partials, values, leaf_weights

    def reduce(query_stats, *, groups=1):
        query_stats = np.ascontiguousarray(query_stats, np.float32)
        assert query_stats.ndim == 2 and query_stats.shape[1] == 2
        partials = np.full((groups, 2), np.nan, np.float32)
        leaf_ids = np.full(len(query_stats), 0xffffffff, np.uint32)
        error = ct.create_string_buffer(4096)
        code = library.cbm_querywise_objective_probe(query_stats.ctypes.data, len(query_stats), groups,
            partials.ctypes.data, leaf_ids.ctypes.data, error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return partials, leaf_ids

    run.project = project
    run.reduce = reduce

    def validate_curvature(curvature):
        curvature = np.ascontiguousarray(curvature, np.float32)
        flag = ct.c_uint32(0xffffffff)  # The probe mirrors runtime's GPU clear.
        error = ct.create_string_buffer(4096)
        code = library.cbm_querywise_curvature_probe(curvature.ctypes.data, len(curvature),
            ct.byref(flag), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return flag.value

    run.validate_curvature = validate_curvature
    return run


def reference(objective, targets, weights, point, offsets, *, beta=1, lambda_reg=.01):
    """Independent double-precision equations, evaluated from float32 inputs."""
    targets, weights, point = (np.asarray(value, np.float32).astype(np.float64) for value in (targets, weights, point))
    beta, lambda_reg = float(np.float32(beta)), float(np.float32(lambda_reg))
    gradient, curvature = np.zeros_like(point), np.zeros_like(point)
    stats = []
    for begin, end in zip(offsets[:-1], offsets[1:]):
        y, w, a = targets[begin:end], weights[begin:end], point[begin:end]
        if objective == 12:
            residual = y - a
            mean = np.dot(w, residual) / w.sum() if w.sum() else 0
            gradient[begin:end] = w * (residual - mean)
            curvature[begin:end] = w
            stats.append([np.dot(w, (residual - mean) ** 2), w.sum()])
        else:
            active = w > 0
            mass = np.dot(y, w)
            if not active.any() or mass == 0:
                stats.append([0, mass])
                continue
            logits = beta * a[active] + np.log(w[active])
            shifted = logits - logits.max()
            log_prob = shifted - np.log(np.exp(shifted).sum())
            probability = np.zeros_like(w)
            probability[active] = np.exp(log_prob)
            gradient[begin:end] = beta * (w * y - mass * probability)
            curvature[begin:end][active] = beta * mass * (beta * probability[active] * (1 - probability[active]) + lambda_reg)
            stats.append([-np.dot(w[active] * y[active], log_prob), mass])
    return gradient, curvature, np.asarray(stats)


def fixture(sizes=(1, 2, 17, 255, 256, 257, 1031, 8193), seed=8520):
    rng = np.random.default_rng(seed)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.uint32)
    rows = int(offsets[-1])
    targets = rng.uniform(0, 4, rows).astype(np.float32)
    weights = rng.uniform(.01, 5, rows).astype(np.float32)
    weights[::7] = 0
    point = rng.normal(0, 2, rows).astype(np.float32)
    return targets, weights, point, offsets


@pytest.mark.parametrize("objective", [12, 13])
def test_mixed_queries_match_cuda_equations(query_probe, objective):
    args = fixture()
    actual = query_probe(objective, *args)
    expected = reference(objective, *args)
    np.testing.assert_array_equal(actual[0], args[2])
    for measured, wanted in zip(actual[1:], expected):
        np.testing.assert_allclose(measured, wanted, rtol=2e-5, atol=2e-4)
    for begin, end in zip(args[3][:-1], args[3][1:]):
        assert abs(actual[1][begin:end].sum(dtype=np.float64)) < 2e-3 * max(1, end - begin)


@pytest.mark.parametrize("beta", [-2, 0, .125, 1, 3])
@pytest.mark.parametrize("lambda_reg", [-.03, 0, .01, .3])
def test_softmax_beta_and_lambda_equations(query_probe, beta, lambda_reg):
    args = fixture((2, 5, 33))
    actual = query_probe(13, *args, beta=beta, lambda_reg=lambda_reg)
    expected = reference(13, *args, beta=beta, lambda_reg=lambda_reg)
    for measured, wanted in zip(actual[1:], expected):
        np.testing.assert_allclose(measured, wanted, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("objective", [12, 13])
def test_zero_weight_zero_signal_and_singleton_queries(query_probe, objective):
    targets = np.array([3, 4, 2, 0, 0, 2, 0, 1], np.float32)
    weights = np.array([2, 0, 0, 1, 2, 0, 1, 2], np.float32)
    point = np.array([4, 1e4, -1e4, 3, -4, 1e4, 0, 1], np.float32)
    offsets = np.array([0, 1, 3, 6, 8], np.uint32)
    actual = query_probe(objective, targets, weights, point, offsets)
    expected = reference(objective, targets, weights, point, offsets)
    for measured, wanted in zip(actual[1:], expected):
        np.testing.assert_allclose(measured, wanted, rtol=2e-6, atol=1e-5)
        assert np.isfinite(measured).all()
    assert actual[1][0] == 0
    assert actual[2][0] == pytest.approx(2 if objective == 12 else .06)
    assert actual[3][0, 0] == 0
    np.testing.assert_array_equal(actual[1][weights == 0], 0)
    np.testing.assert_array_equal(actual[2][weights == 0], 0)


@pytest.mark.parametrize("objective", [12, 13])
def test_query_shift_invariance(query_probe, objective):
    # Exactly representable points make float32 input rounding independent of
    # the invariance being tested, including large common prediction offsets.
    targets, weights, point, offsets = fixture((17, 271, 9))
    point = (np.round(point * 4) / 4).astype(np.float32)
    shifted = point.copy()
    for delta, begin, end in zip([2**20, -2**18, 128], offsets[:-1], offsets[1:]):
        shifted[begin:end] += delta
    original = query_probe(objective, targets, weights, point, offsets, beta=.375)
    moved = query_probe(objective, targets, weights, shifted, offsets, beta=.375)
    # RMSE y-a must itself be representable; choose integer y at large shifts.
    if objective == 12:
        targets = np.round(targets)
        original = query_probe(objective, targets, weights, point, offsets)
        moved = query_probe(objective, targets, weights, shifted, offsets)
        # Centering two large float32 residuals incurs one float32 ULP.
        np.testing.assert_allclose(moved[1], original[1], rtol=.04, atol=.4)
        np.testing.assert_allclose(moved[3], original[3], rtol=.004, atol=.05)
    else:
        for measured, wanted in zip(moved[1:], original[1:]):
            np.testing.assert_array_equal(measured, wanted)


def test_softmax_extreme_points_have_finite_log_domain_loss(query_probe):
    args = (np.array([2, 1, 3, 7], np.float32), np.array([1, 2, 3, 0], np.float32),
            np.array([-1e4, 1e4, -5000, 1e8], np.float32), np.array([0, 4], np.uint32))
    actual = query_probe(13, *args)
    expected = reference(13, *args)
    for measured, wanted in zip(actual[1:], expected):
        assert np.isfinite(measured).all()
        np.testing.assert_allclose(measured, wanted, rtol=2e-6, atol=1e-4)


@pytest.mark.parametrize("objective", [12, 13])
def test_cross_leaf_trial_recomputes_complete_queries(query_probe, objective):
    targets, weights, cursor, offsets = fixture((19, 263, 517))
    ids = np.arange(len(cursor), dtype=np.uint32) % 8
    leaves = np.array([.7, -.3, 2, -.9, .12, 1.2, -.6, 0], np.float32)
    actual = query_probe(objective, targets, weights, cursor, offsets, leaf_values=leaves, leaf_ids=ids)
    expected_point = (cursor + leaves[ids]).astype(np.float32)
    np.testing.assert_array_equal(actual[0], expected_point)
    expected = reference(objective, targets, weights, expected_point, offsets)
    for measured, wanted in zip(actual[1:], expected):
        np.testing.assert_allclose(measured, wanted, rtol=2e-5, atol=2e-4)
    # Projection is downstream of query normalization, for both leaf methods.
    projected = np.bincount(ids, weights=actual[1], minlength=8)
    expected_projected = np.bincount(ids, weights=expected[0], minlength=8)
    np.testing.assert_allclose(projected, expected_projected, rtol=2e-5, atol=1e-3)


@pytest.mark.parametrize("objective", [12, 13])
def test_gradient_matches_query_loss_finite_difference(query_probe, objective):
    args = fixture((7, 11))
    targets, weights, point, offsets = args
    actual = query_probe(objective, *args)
    epsilon = .001
    for row in [1, 4, 9, 15]:
        left, right = point.copy(), point.copy()
        left[row] -= epsilon
        right[row] += epsilon
        loss_left = reference(objective, targets, weights, left, offsets)[2][:, 0].sum()
        loss_right = reference(objective, targets, weights, right, offsets)[2][:, 0].sum()
        derivative = (loss_right - loss_left) / float(right[row] - left[row])
        # CUDA QueryRMSE uses the derivative of half the squared loss.
        expected = -derivative / (2 if objective == 12 else 1)
        assert actual[1][row] == pytest.approx(expected, rel=3e-5, abs=3e-5)


@pytest.mark.parametrize("objective", [12, 13])
def test_long_query_strided_reduction(query_probe, objective):
    args = fixture((65539, 1, 1025))
    actual = query_probe(objective, *args)
    expected = reference(objective, *args)
    for measured, wanted in zip(actual[1:], expected):
        np.testing.assert_allclose(measured, wanted, rtol=3e-5, atol=.01)


def test_softmax_lambda_changes_only_curvature(query_probe):
    args = fixture((31, 9))
    plain = query_probe(13, *args, lambda_reg=0)
    regularized = query_probe(13, *args, lambda_reg=.3)
    np.testing.assert_array_equal(plain[1], regularized[1])
    np.testing.assert_array_equal(plain[3], regularized[3])
    for begin, end in zip(args[3][:-1], args[3][1:]):
        mass = np.dot(args[0][begin:end].astype(np.float64), args[1][begin:end])
        expected = np.where(args[1][begin:end] > 0, mass * np.float32(.3), 0)
        np.testing.assert_allclose(regularized[2][begin:end] - plain[2][begin:end], expected, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("offsets", [[1, 5], [0, 4], [0, 3, 2, 5], [0, 0, 5]])
def test_invalid_query_offsets_are_rejected(query_probe, offsets):
    with pytest.raises(ValueError, match="Query offsets"):
        query_probe(12, np.ones(5), np.ones(5), np.zeros(5), offsets)


@pytest.mark.parametrize("small", [1, 2**-20])
@pytest.mark.parametrize("positions,rows", [([0, 1, 2], 3), ([0, 256, 512], 769)])
def test_rmse_query_mean_preserves_cancellation(query_probe, small, positions, rows):
    targets = np.zeros(rows, np.float32)
    targets[positions] = [2**24, small, -(2**24)]
    actual = query_probe(12, targets, np.ones(rows), np.zeros(rows), [0, rows])
    expected_mean = small / rows
    # Positive signal can be many orders below the cancelled residuals. Both
    # cross-lane reduction and repeated additions in one lane must preserve it.
    assert actual[1][positions[1]] > 0
    assert actual[1][positions[1]] == pytest.approx(small - expected_mean, rel=2e-7, abs=0)
    if rows > 3:
        assert actual[1][1] == pytest.approx(-expected_mean, rel=2e-7, abs=0)


@pytest.mark.parametrize("objective", [12, 13])
def test_zero_weight_row_skips_overflowing_residual(query_probe, objective):
    maximum = np.finfo(np.float32).max
    args = ([2, 1, maximum], [1, 2, 0], [0, 1, -maximum], [0, 3])
    actual = query_probe(objective, *args)
    expected = reference(objective, *args)
    for measured, wanted in zip(actual[1:], expected):
        assert np.isfinite(measured).all()
        np.testing.assert_allclose(measured, wanted, rtol=2e-6, atol=1e-6)
    assert actual[1][-1] == actual[2][-1] == 0


@pytest.mark.parametrize("tiles", [1, 3, 257])
@pytest.mark.parametrize("leaf_method", [0, 1])
def test_gpu_projector_matches_complete_query_projection_and_scalar_solver(query_probe, tiles, leaf_method):
    args = fixture((19, 263, 1027))
    _, gradient, curvature, _ = query_probe(13, *args)
    ids = np.arange(len(gradient), dtype=np.uint32) % 5
    rows = np.argsort(ids, kind="stable").astype(np.uint32)
    counts = np.bincount(ids, minlength=6)  # Last leaf is empty.
    offsets = np.r_[0, np.cumsum(counts)].astype(np.uint32)
    partials, values, weights = query_probe.project(gradient, curvature, args[1], rows, offsets,
        tiles=tiles, leaf_method=leaf_method, l2=2.5)
    expected = np.column_stack([np.bincount(ids, weights=data, minlength=6)
        for data in (gradient, curvature if leaf_method == 0 else np.zeros_like(curvature), args[1])])
    actual = partials.astype(np.float64).sum(axis=(1, 2))[:, :3]
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-12)
    np.testing.assert_array_equal(partials[:, :, :, 3], 0)
    np.testing.assert_allclose(weights, expected[:, 2], rtol=1e-7, atol=0)
    diagonal = expected[:, 2 if leaf_method == 1 else 1] + 2.5
    np.testing.assert_allclose(values, expected[:, 0] / diagonal, rtol=2e-7, atol=1e-7)
    assert values[-1] == weights[-1] == 0


@pytest.mark.parametrize("tiles", [1, 2, 3])
def test_gpu_projector_keeps_cancellation_through_leaf_tiles(query_probe, tiles):
    rows_count = 1545
    ids = np.arange(rows_count, dtype=np.uint32) % 3
    rows = np.argsort(ids, kind="stable").astype(np.uint32)
    offsets = np.r_[0, np.cumsum(np.bincount(ids, minlength=4))].astype(np.uint32)
    gradient = np.zeros(rows_count, np.float32)
    # Leaf 0 cancels across lanes, leaf 1 across 256-row tiles/strided loads.
    gradient[rows[[0, 1, 2]]] = [2**24, 1, -(2**24)]
    base = offsets[1]
    gradient[rows[base + np.array([0, 256, 512])]] = [2**24, 2**-20, -(2**24)]
    partials, values, weights = query_probe.project(gradient, np.ones(rows_count), np.ones(rows_count),
        rows, offsets, tiles=tiles, l2=0)
    observed = partials[:, :, :, 0].astype(np.float64).sum(axis=(1, 2))
    np.testing.assert_array_equal(observed, [1, 2**-20, 0, 0])
    assert values[0] == pytest.approx(1 / 515, rel=1e-7)
    assert values[1] == pytest.approx(2**-20 / 515, rel=1e-7)
    np.testing.assert_array_equal(weights, [515, 515, 515, 0])


@pytest.mark.parametrize("unused_hessian", [-7, np.inf, np.nan])
def test_gradient_projection_ignores_unused_curvature(query_probe, unused_hessian):
    partials, values, weights = query_probe.project([1, 2, 3], [unused_hessian] * 3, [1, 2, 3],
        [2, 0, 1], [0, 2, 3], leaf_method=1)
    assert np.isfinite(partials).all()
    np.testing.assert_array_equal(partials[:, :, :, 1], 0)
    np.testing.assert_allclose(values, [4 / 6, 2 / 4], rtol=1e-7)
    np.testing.assert_array_equal(weights, [4, 2])


@pytest.mark.parametrize("beta,lambda_reg", [(1, -1), (1e30, 1e30)])
def test_newton_structure_curvature_validation_is_opt_in(query_probe, beta, lambda_reg):
    args = ([1, 1], [1, 1], [0, 0], [0, 2])
    plain = query_probe(13, *args, beta=beta, lambda_reg=lambda_reg)
    guarded = query_probe(13, *args, beta=beta, lambda_reg=lambda_reg, reserved=1)
    np.testing.assert_array_equal(plain[1], 0)
    assert np.isnan(guarded[1]).all()
    np.testing.assert_array_equal(guarded[2], plain[2])
    np.testing.assert_array_equal(guarded[3], plain[3])
    partials, values, weights = query_probe.project(plain[1], plain[2], args[1], [0, 1], [0, 2],
        leaf_method=1)
    assert np.isfinite(partials).all()
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(weights, 2)


@pytest.mark.parametrize("groups", [1, 7, 4096])
def test_bounded_query_objective_reduction_and_leaf_id_reset(query_probe, groups):
    query_count = 65539
    rng = np.random.default_rng(2347)
    stats = rng.uniform(0, 32, (query_count, 2)).astype(np.float32)
    partials, leaf_ids = query_probe.reduce(stats, groups=groups)
    assert partials.shape == (groups, 2)
    np.testing.assert_array_equal(leaf_ids, 0)
    # Check each strided block, including untouched full groups at the tail.
    expected = np.zeros_like(partials, dtype=np.float64)
    assignment = np.arange(query_count) // 256 % groups
    for component in range(2):
        expected[:, component] = np.bincount(assignment, weights=stats[:, component], minlength=groups)
    np.testing.assert_array_equal(partials, expected.astype(np.float32))


def test_query_objective_reduction_compensation(query_probe):
    stats = np.zeros((769, 2), np.float32)
    stats[[0, 256, 512], 0] = [2**24, 1, -(2**24)]
    stats[[0, 1, 2], 1] = [2**24, 2**-20, -(2**24)]
    partials, _ = query_probe.reduce(stats)
    np.testing.assert_array_equal(partials, [[1, 2**-20]])


@pytest.mark.parametrize("value,expected", [(0, 0), (-0., 0), (3.5, 0), (-1e-20, 1),
    (np.nan, 1), (np.inf, 1), (-np.inf, 1)])
@pytest.mark.parametrize("row", [0, 128, 256, 1026])
def test_explicit_structure_curvature_flag(query_probe, value, expected, row):
    curvature = np.ones(1027, np.float32)
    curvature[row] = value
    assert query_probe.validate_curvature(curvature) == expected
