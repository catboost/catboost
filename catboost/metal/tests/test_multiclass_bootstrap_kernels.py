"""Actual GPU multiclass sampling and greedy-search variance, from CUDA math."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest

from test_bootstrap import probe as bootstrap_probe


class MulticlassBootstrapParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("rows", "dimensions", "multi_logit", "reserved")]


@pytest.fixture(scope="module")
def multiclass_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("multiclass_bootstrap_probe.mm")
    root = source.parent.parent
    files = [source, root / "native/metal_bootstrap_kernels.h", root / "native/metal_multiclass_bootstrap.h"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in files)).hexdigest()[:16]
    destination = root / ".build" / f"multiclass_bootstrap_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_multiclass_bootstrap_probe.argtypes = [ct.POINTER(MulticlassBootstrapParams), ct.c_uint32] + [ct.c_void_p] * 7 + [ct.c_uint32]
    library.cbm_multiclass_bootstrap_probe.restype = ct.c_int

    def run(gradients, weights, multipliers=None, *, multi_logit=True, groups=7):
        gradients = np.ascontiguousarray(gradients, np.float32)
        dimensions, rows = gradients.shape
        weights = np.ascontiguousarray(weights, np.float32)
        multipliers = np.ascontiguousarray(np.ones(rows) if multipliers is None else multipliers, np.float32)
        assert weights.shape == multipliers.shape == (rows,)
        sampled = np.zeros_like(gradients)
        sampled_weights = np.zeros_like(weights)
        partials = np.zeros((groups, 2), np.float32)
        error = ct.create_string_buffer(4096)
        p = MulticlassBootstrapParams(rows, dimensions, multi_logit, 0)
        buffers = (gradients, weights, multipliers, sampled, sampled_weights, partials)
        code = library.cbm_multiclass_bootstrap_probe(ct.byref(p), groups,
            *(value.ctypes.data for value in buffers), error, len(error))
        assert code == 0, error.value.decode()
        return sampled, sampled_weights, partials.sum(axis=0, dtype=np.float64)
    return run


def reference_statistics(gradients, weights, multi_logit):
    gradients = np.asarray(gradients, np.float32).astype(np.float64)
    weights = np.asarray(weights, np.float32).astype(np.float64)
    active = weights > float(np.float32(1e-15))
    complete = gradients
    if multi_logit:
        complete = np.concatenate((gradients, -gradients.sum(axis=0, keepdims=True)))
    numerator = (complete[:, active] ** 2 / weights[active]).sum()
    denominator = weights[active].sum()
    return np.array([numerator, denominator]) / len(weights)


def observations(dimensions, rows, multi_logit):
    rng = np.random.default_rng(974 + dimensions + rows)
    classes = dimensions + int(multi_logit)
    logits = rng.normal(0, 2, (classes, rows))
    if multi_logit:
        exponentials = np.exp(logits - logits.max(axis=0))
        probabilities = exponentials / exponentials.sum(axis=0)
    else:
        probabilities = 1 / (1 + np.exp(-logits))
    labels = rng.integers(0, classes, rows)
    weights = rng.uniform(.1, 5, rows).astype(np.float32)
    weights[::11] = 0
    gradients = (weights * ((np.arange(classes)[:, None] == labels) - probabilities))[:dimensions].astype(np.float32)
    return gradients, weights


@pytest.mark.parametrize("dimensions", [1, 2, 7, 63])
@pytest.mark.parametrize("rows", [1, 259, 8197])
@pytest.mark.parametrize("multi_logit", [False, True])
def test_multiclass_statistic_matches_cuda_equation(multiclass_probe, dimensions, rows, multi_logit):
    gradients, weights = observations(dimensions, rows, multi_logit)
    sampled, sampled_weights, statistic = multiclass_probe(gradients, weights, multi_logit=multi_logit)
    np.testing.assert_array_equal(sampled, gradients)
    np.testing.assert_array_equal(sampled_weights, weights)
    np.testing.assert_allclose(statistic, reference_statistics(gradients, weights, multi_logit), rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize("multi_logit", [False, True])
def test_real_gpu_samplers_share_one_object_draw(multiclass_probe, bootstrap_probe, kind, multi_logit):
    gradients, weights = observations(7, 1031, multi_logit)
    multipliers = bootstrap_probe(kind, rows=1031, seed=582, iteration=17, temperature=.7, subsample=.37)["values"]
    sampled, sampled_weights, statistic = multiclass_probe(gradients, weights, multipliers, multi_logit=multi_logit)
    expected_g = (gradients * multipliers).astype(np.float32)
    expected_w = (weights * multipliers).astype(np.float32)
    np.testing.assert_array_equal(sampled, expected_g)
    np.testing.assert_array_equal(sampled_weights, expected_w)
    np.testing.assert_allclose(statistic, reference_statistics(expected_g, expected_w, multi_logit), rtol=2e-6, atol=1e-7)
    if kind:
        # Statistics must follow the sampled target, not the original target.
        assert not np.allclose(statistic, reference_statistics(gradients, weights, multi_logit))


def test_missing_class_contributes_without_class_average(multiclass_probe):
    gradients = np.array([[1, 2], [2, 3]], np.float32)
    weights = np.array([2, 4], np.float32)
    mc = multiclass_probe(gradients, weights, multi_logit=True)[2]
    ova = multiclass_probe(gradients, weights, multi_logit=False)[2]
    np.testing.assert_allclose(mc, [(14 / 2 + 38 / 4) / 2, 3], atol=1e-6)
    np.testing.assert_allclose(ova, [(5 / 2 + 13 / 4) / 2, 3], atol=1e-6)


@pytest.mark.parametrize("multi_logit", [False, True])
def test_strict_weight_threshold_without_gradient_zeroing(multiclass_probe, multi_logit):
    boundary = np.float32(1e-15)
    weights = np.array([0, np.nextafter(boundary, np.float32(0)), boundary,
                       np.nextafter(boundary, np.float32(np.inf)), 1], np.float32)
    gradients = np.array([[1e10, 1e10, 1e10, 1e-16, 2], [1e10, 1e10, 1e10, 2e-16, -1]], np.float32)
    result = multiclass_probe(gradients, weights, multi_logit=multi_logit)[2]
    np.testing.assert_allclose(result, reference_statistics(gradients, weights, multi_logit), rtol=2e-6, atol=1e-7)
    # Unlike scalar ZeroAwareDivide, small nonzero gradients are not discarded.
    tiny = multiclass_probe(gradients[:, 3:4], weights[3:4], multi_logit=multi_logit)[2]
    assert tiny[0] > 0


def test_excluded_rows_mask_every_class_and_nonfinite_gradients(multiclass_probe):
    gradients = np.array([[np.inf, -np.inf, 2], [np.nan, np.inf, 3]], np.float32)
    weights = np.array([1, 2, 3], np.float32)
    sampled, sampled_weights, statistic = multiclass_probe(gradients, weights, [0, 0, 1])
    np.testing.assert_array_equal(sampled[:, :2], 0)
    np.testing.assert_array_equal(sampled_weights[:2], 0)
    np.testing.assert_allclose(statistic, reference_statistics(sampled, sampled_weights, True), rtol=2e-6)


@pytest.mark.parametrize("multi_logit", [False, True])
def test_uniform_bootstrap_scale_preserves_stddev(multiclass_probe, multi_logit):
    gradients, weights = observations(7, 1031, multi_logit)
    original = multiclass_probe(gradients, weights, multi_logit=multi_logit)[2]
    scaled = multiclass_probe(gradients, weights, np.full(1031, 4), multi_logit=multi_logit)[2]
    np.testing.assert_allclose(scaled, original * 4, rtol=2e-6)
    assert scaled[0] / scaled[1] == pytest.approx(original[0] / original[1], rel=2e-6)


def test_missing_dimension_choice_preserves_multiclass_statistic(multiclass_probe):
    gradients, weights = observations(7, 1031, True)
    complete = np.concatenate((gradients, -gradients.sum(axis=0, keepdims=True)))
    expected = multiclass_probe(gradients, weights)[2]
    for omitted in [0, 3, 7]:
        selected = np.delete(complete, omitted, axis=0)
        np.testing.assert_allclose(multiclass_probe(selected, weights)[2], expected, rtol=2e-6)


@pytest.mark.parametrize("groups", [1, 3, 257])
def test_strided_reduction_launch_independence(multiclass_probe, groups):
    gradients, weights = observations(3, 65539, True)
    result = multiclass_probe(gradients, weights, groups=groups)[2]
    np.testing.assert_allclose(result, reference_statistics(gradients, weights, True), rtol=2e-6)


def test_large_finite_weights_avoid_square_overflow(multiclass_probe):
    gradients, weights = observations(7, 1031, True)
    gradients *= np.float32(1e30)
    weights *= np.float32(1e30)
    result = multiclass_probe(gradients, weights)[2]
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, reference_statistics(gradients, weights, True), rtol=3e-6)


def test_all_excluded_target_has_zero_noise_statistics(multiclass_probe):
    gradients, weights = observations(7, 1031, True)
    sampled, sampled_weights, statistic = multiclass_probe(gradients, weights, np.zeros(1031))
    np.testing.assert_array_equal(sampled, 0)
    np.testing.assert_array_equal(sampled_weights, 0)
    np.testing.assert_array_equal(statistic, 0)
