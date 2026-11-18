"""GPU coupled leaf matrices versus independent NumPy linear algebra."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class MatrixParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("leaves", "has_diagonal_part", "reserved0", "reserved1")]
    _fields_ += [(name, ct.c_float) for name in ("l2", "non_diag_l2", "min_leaf_weight", "step")]


@pytest.fixture(scope="module")
def matrix_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("leaf_matrix_probe.mm")
    root = source.parent.parent
    header = root / "native/metal_leaf_matrix_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:16]
    destination = root / ".build" / f"leaf_matrix_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_leaf_matrix_probe.argtypes = [ct.POINTER(MatrixParams)] + [ct.c_void_p] * 10 + [ct.c_uint32]
    library.cbm_leaf_matrix_probe.restype = ct.c_int

    def run(hessian, gradient, *, diagonal=False, l2=3, non_diag_l2=.2,
            weights=None, point=None, minimum=1e-20, step=1):
        hessian, gradient = np.ascontiguousarray(hessian, np.float32), np.ascontiguousarray(gradient, np.float32)
        n = gradient.size
        assert hessian.shape == (n, n)
        weights = np.ascontiguousarray(np.ones(n) if weights is None else weights, np.float32)
        point = np.ascontiguousarray(np.zeros(n) if point is None else point, np.float32)
        regularized = np.zeros((n, n, 2), np.float32)
        direction, updated, dot = np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(2, np.float32)
        status, error = ct.c_uint32(), ct.create_string_buffer(4096)
        p = MatrixParams(n, diagonal, 0, 0, l2, non_diag_l2, minimum, step)
        arrays = hessian, gradient, weights, point, regularized, direction, updated, dot
        code = library.cbm_leaf_matrix_probe(ct.byref(p), *(a.ctypes.data for a in arrays),
                                             ct.byref(status), error, len(error))
        if code:
            raise ValueError(error.value.decode())
        return dict(regularized=regularized.sum(axis=-1, dtype=np.float64), expansion=regularized,
                    direction=direction, updated=updated, dot=dot.sum(dtype=np.float64), status=status.value)
    return run


def regularize(hessian, diagonal, l2, non_diag_l2):
    hessian = np.asarray(hessian, np.float32).astype(np.float64)
    leaves = hessian.shape[0]
    n = leaves if diagonal else leaves - 1
    result = np.zeros_like(hessian)
    if n:
        result[:n, :n] = hessian[:n, :n] - float(np.float32(non_diag_l2)) / leaves
        ids = np.arange(n)
        result[ids, ids] += (float(np.float32(non_diag_l2)) + float(np.float32(l2))
                              + 10 * (hessian[ids, ids] == 0))
    return result


def laplacian(leaves, seed=8921):
    rng = np.random.default_rng(seed + leaves)
    edges = rng.uniform(.01, 1, (leaves, leaves))
    edges = np.triu(edges, 1)
    edges += edges.T
    return (np.diag(edges.sum(axis=1)) - edges).astype(np.float32)


@pytest.mark.parametrize("leaves", [1, 2, 7, 31, 64, 128, 256])
@pytest.mark.parametrize("diagonal", [False, True])
def test_coupled_leaf_direction_matches_full_numpy_system(matrix_probe, leaves, diagonal):
    hessian = laplacian(leaves)
    if diagonal:
        hessian += np.eye(leaves, dtype=np.float32) * .1
    gradient = np.random.default_rng(819 + leaves).normal(size=leaves).astype(np.float32)
    result = matrix_probe(hessian, gradient, diagonal=diagonal)
    assert result["status"] == 0
    matrix = regularize(hessian, diagonal, 3, .2)
    np.testing.assert_allclose(result["regularized"], matrix, rtol=2e-14, atol=2e-13)
    n = leaves if diagonal else leaves - 1
    expected = np.zeros(leaves)
    if n:
        expected[:n] = np.linalg.solve(matrix[:n, :n], gradient[:n])
    np.testing.assert_allclose(result["direction"], expected, rtol=3e-6, atol=3e-7)
    np.testing.assert_array_equal(result["updated"], result["direction"])
    assert result["dot"] == pytest.approx(gradient.astype(float) @ result["direction"].astype(float), rel=2e-12, abs=2e-13)


@pytest.mark.parametrize("diagonal", [False, True])
@pytest.mark.parametrize("l2,non_diag", [(0, 0), (3, 0), (0, 2), (.25, .75)])
def test_zero_original_diagonal_gets_ten_before_regularization(matrix_probe, diagonal, l2, non_diag):
    hessian, gradient = np.zeros((4, 4), np.float32), np.array([1, -2, 3, -4], np.float32)
    result = matrix_probe(hessian, gradient, diagonal=diagonal, l2=l2, non_diag_l2=non_diag)
    matrix = regularize(hessian, diagonal, l2, non_diag)
    n = 4 if diagonal else 3
    assert result["status"] == 0
    np.testing.assert_array_equal(result["regularized"], matrix)
    np.testing.assert_allclose(result["direction"][:n], np.linalg.solve(matrix[:n, :n], gradient[:n]), rtol=2e-6)


@pytest.mark.parametrize("exponent", [20, 25, 30])
@pytest.mark.parametrize("diagonal", [False, True])
def test_sub_ulp_ridge_survives_disconnected_large_laplacians(matrix_probe, exponent, diagonal):
    scale = np.float32(2. ** exponent)
    hessian = np.zeros((4, 4), np.float32)
    hessian[:2, :2] = hessian[2:, 2:] = [[scale, -scale], [-scale, scale]]
    gradient = np.array([1, 2, -2, 3], np.float32)
    result = matrix_probe(hessian, gradient, diagonal=diagonal, l2=.25, non_diag_l2=0)
    matrix = regularize(hessian, diagonal, .25, 0)
    n = 4 if diagonal else 3
    assert result["status"] == 0
    assert result["regularized"][0, 0] - float(scale) == .25
    np.testing.assert_allclose(result["direction"][:n], np.linalg.solve(matrix[:n, :n], gradient[:n]), rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("diagonal", [False, True])
def test_mask_is_applied_to_updated_point_after_coupled_solve(matrix_probe, diagonal):
    hessian, gradient = laplacian(7), np.arange(-3, 4, dtype=np.float32)
    point = np.array([1, 0, -2, 4, 1, 3, 8], np.float32)
    weights = np.array([1, 0, .1, 2, 0, 1, 1], np.float32)
    result = matrix_probe(hessian, gradient, diagonal=diagonal, weights=weights, point=point, minimum=.2, step=.125)
    assert result["status"] == 0
    expected = (point.astype(float) + .125 * result["direction"].astype(float)).astype(np.float32)
    expected[weights < np.float32(.2)] = 0
    if not diagonal:
        expected[-1] = 0
    np.testing.assert_array_equal(result["updated"], expected)
    assert result["direction"][1] != 0  # The empty leaf still participates in the matrix solve.


@pytest.mark.parametrize("kind", ["asymmetry", "nan_hessian", "inf_hessian", "nan_gradient", "negative_weight", "indefinite"])
def test_invalid_coupled_system_sets_gpu_status(matrix_probe, kind):
    hessian, gradient, weights = np.eye(3, dtype=np.float32), np.ones(3, np.float32), np.ones(3, np.float32)
    if kind == "asymmetry": hessian[0, 1] = 1
    if kind == "nan_hessian": hessian[0, 0] = np.nan
    if kind == "inf_hessian": hessian[1, 0] = hessian[0, 1] = np.inf
    if kind == "nan_gradient": gradient[1] = np.nan
    if kind == "negative_weight": weights[0] = -1
    if kind == "indefinite": hessian[0, 0] = -10
    result = matrix_probe(hessian, gradient, weights=weights, diagonal=True, l2=0, non_diag_l2=0)
    assert result["status"] & (2 if kind == "indefinite" else 1)


def test_gpu_solver_repeats_exactly(matrix_probe):
    args = laplacian(64), np.random.default_rng(418).normal(size=64).astype(np.float32)
    first, second = matrix_probe(*args), matrix_probe(*args)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


@pytest.mark.parametrize("options", [dict(l2=-1), dict(non_diag_l2=-1), dict(l2=np.inf), dict(step=np.nan)])
def test_invalid_leaf_matrix_configuration_is_rejected(matrix_probe, options):
    with pytest.raises(ValueError, match="parameters"):
        matrix_probe(np.eye(2), np.ones(2), **options)
