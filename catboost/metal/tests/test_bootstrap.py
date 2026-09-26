"""CUDA-derived bootstrap kernels exercised on the actual Apple GPU.

The scalar RNG is an independent integer reference; no CPU CatBoost training
is involved. Distribution checks have wide deterministic statistical bounds.
"""

import ctypes as ct
import hashlib
import math
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class BootstrapParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "type", "seed_low", "seed_high", "iteration", "stream", "reserved0", "reserved1"
    )] + [(name, ct.c_float) for name in ("temperature", "subsample", "mvs_lambda", "noise_scale")]


@pytest.fixture(scope="module")
def probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).with_name("bootstrap_probe.mm")
    header = root / "native/metal_bootstrap_kernels.h"
    noise_header = root / "native/metal_score_noise_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes() + noise_header.read_bytes()).hexdigest()[:16]
    destination = root / ".build" / f"bootstrap_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_bootstrap_probe.argtypes = [ct.POINTER(BootstrapParams)] + [ct.c_void_p] * 8 + [ct.c_uint32]
    library.cbm_bootstrap_probe.restype = ct.c_int

    def run(kind, rows=1031, *, seed=72831, iteration=0, stream=0, temperature=1.0,
            subsample=.66, mvs_lambda=0.0, noise_scale=1.0, derivatives=None, weights=None):
        gradients = np.asarray(np.ones(rows) if derivatives is None else derivatives, dtype=np.float32)
        sample_weights = np.asarray(np.ones(rows) if weights is None else weights, dtype=np.float32)
        assert gradients.shape == sample_weights.shape == (rows,)
        output = np.zeros(rows, np.float32)
        scaled = np.zeros_like(output)
        structure = np.zeros_like(output)
        thresholds = np.zeros((rows + 8191) // 8192, np.float32)
        words = np.zeros(rows, np.uint32)
        params = BootstrapParams(rows, kind, seed & 0xffffffff, seed >> 32, iteration, stream,
                                 0, 0, temperature, subsample, mvs_lambda, noise_scale)
        error = ct.create_string_buffer(4096)
        code = library.cbm_bootstrap_probe(ct.byref(params), gradients.ctypes.data,
            sample_weights.ctypes.data, output.ctypes.data, scaled.ctypes.data,
            structure.ctypes.data, thresholds.ctypes.data, words.ctypes.data, error, len(error))
        assert code == 0, error.value.decode()
        return dict(values=output, gradients=scaled, structure_weights=structure,
                    thresholds=thresholds, words=words)
    return run


MASK = (1 << 64) - 1


def mix_seed(value):
    value = (value + 0x9e3779b97f4a7c15) & MASK
    value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK
    value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK
    return value ^ (value >> 31)


def seed_for_item(item, seed=72831, iteration=0, stream=0):
    value = mix_seed(seed ^ mix_seed(iteration ^ 0xd1b54a32d192ed03)
                     ^ mix_seed(((stream << 32) | item) ^ 0x94d049bb133111eb))
    if value & 0xffffffff == 0:
        value |= 1
    if value >> 32 == 0:
        value |= 1 << 32
    return value


def next_word(seed):
    v, u = seed >> 32, seed & 0xffffffff
    v = (36969 * (v & 0xffff) + (v >> 16)) & 0xffffffff
    u = (18000 * (u & 0xffff) + (u >> 16)) & 0xffffffff
    return (v << 32) | u, ((v << 16) + u) & 0xffffffff


def uniforms(rows, **kwargs):
    words = np.array([next_word(seed_for_item(i, **kwargs))[1] for i in range(rows)], dtype=np.uint32)
    return np.minimum(words.astype(np.float32) * np.float32(2.0**-32),
                      np.nextafter(np.float32(1), np.float32(0)))


def test_integer_rng_matches_reference_and_stream_identity(probe):
    args = dict(seed=0xfedcba9876543210, iteration=729, stream=7)
    result = probe(5, **args)
    expected = [next_word(seed_for_item(i, **args))[1] for i in range(1031)]
    np.testing.assert_array_equal(result["words"], expected)
    np.testing.assert_array_equal(result["values"], uniforms(1031, **args))
    np.testing.assert_array_equal(result["values"], probe(5, **args)["values"])
    assert not np.array_equal(result["values"], probe(5, **(args | {"iteration": 730}))["values"])
    # Item/iteration state does not depend on the input length or launch shape.
    np.testing.assert_array_equal(result["values"][:17], probe(5, rows=17, **args)["values"])
    # Domains for absolute iteration and row must not collide when swapped.
    assert probe(5, rows=2, iteration=0)["words"][1] != probe(5, rows=2, iteration=1)["words"][0]


@pytest.mark.parametrize("kind,temperature,subsample", [(0, 1, .66), (1, 0, .66), (2, 1, 1), (4, 1, 1)])
def test_noop_boundaries_preserve_original_weights(probe, kind, temperature, subsample):
    derivatives = np.linspace(-3, 4, 1031, dtype=np.float32)
    weights = np.arange(1031, dtype=np.float32) % 7
    result = probe(kind, temperature=temperature, subsample=subsample,
                   derivatives=derivatives, weights=weights)
    np.testing.assert_array_equal(result["values"], 1)
    np.testing.assert_array_equal(result["gradients"], derivatives)
    np.testing.assert_array_equal(result["structure_weights"], weights)


@pytest.mark.parametrize("temperature", [.25, 1, 2])
def test_bayesian_formula_and_weight_application(probe, temperature):
    derivatives = np.linspace(-7, 3, 1031, dtype=np.float32)
    weights = np.linspace(0, 4, 1031, dtype=np.float32)
    result = probe(1, temperature=temperature, derivatives=derivatives, weights=weights)
    expected = (-np.log(uniforms(1031).astype(np.float64) + 1e-20)) ** temperature
    np.testing.assert_allclose(result["values"], expected, rtol=3e-6, atol=1e-7)
    np.testing.assert_allclose(result["gradients"], derivatives * expected, rtol=4e-6, atol=1e-7)
    np.testing.assert_allclose(result["structure_weights"], weights * expected, rtol=4e-6, atol=1e-7)


def test_bernoulli_exact_seeded_reference_and_zero_boundary(probe):
    result = probe(2, subsample=.31)
    np.testing.assert_array_equal(result["values"], uniforms(1031) < np.float32(.31))
    # Public API rejects subsample=0; the kernel itself must have exact edges.
    np.testing.assert_array_equal(probe(2, subsample=0)["values"], 0)


def test_distribution_invariants(probe):
    rows = 131071
    bernoulli = probe(2, rows, subsample=.31)["values"]
    assert abs(bernoulli.mean() - .31) < .005
    bayesian = probe(1, rows)["values"]
    assert np.all(bayesian >= 0) and np.isfinite(bayesian).all()
    assert abs(bayesian.mean() - 1) < .02
    assert abs(bayesian.var() - 1) < .05
    poisson = probe(3, rows, subsample=.66)["values"]
    rate = -math.log(1 - .66)
    assert np.all(poisson == np.floor(poisson)) and np.all(poisson >= 0)
    assert abs(poisson.mean() - rate) < .02
    assert abs(poisson.var() - rate) < .04
    assert abs(np.count_nonzero(poisson) / rows - .66) < .005
    normals = probe(6, rows)["values"]
    assert np.isfinite(normals).all()
    assert abs(normals.mean()) < .015
    assert abs(normals.var() - 1) < .035


def test_poisson_seeded_log_product_reference(probe):
    expected = []
    rate = np.float32(-math.log(1 - np.float32(.66)))
    for row in range(1031):
        state = seed_for_item(row)
        log_probability = np.float32(0)
        count = 0
        while True:
            state, word = next_word(state)
            uniform = min(np.float32(word) * np.float32(2**-32), np.nextafter(np.float32(1), np.float32(0)))
            log_probability += np.float32(math.log(max(uniform, np.float32(2**-32))))
            count += 1
            if log_probability <= -rate:
                break
        expected.append(count - 1)
    np.testing.assert_array_equal(probe(3)["values"], expected)
    np.testing.assert_array_equal(probe(3, subsample=0)["values"], 0)


def test_normal_transform_matches_seeded_box_muller_reference(probe):
    expected = []
    for row in range(1031):
        state = seed_for_item(row)
        for _ in range(4):
            state, _ = next_word(state)
        state, first = next_word(state)
        _, second = next_word(state)
        a = max(min(np.float32(first) * np.float32(2**-32), np.nextafter(np.float32(1), np.float32(0))),
                np.float32(2**-32))
        b = min(np.float32(second) * np.float32(2**-32), np.nextafter(np.float32(1), np.float32(0)))
        expected.append(math.sqrt(-2 * math.log(a)) * math.cos(float(np.float32(2 * math.pi)) * b))
    np.testing.assert_allclose(probe(6)["values"], expected, rtol=3e-5, atol=2e-6)
    np.testing.assert_array_equal(probe(6, noise_scale=0)["values"], 0)


def _mvs_threshold(values, subsample):
    ordered = np.sort(values.astype(np.float64))
    target = len(ordered) * subsample
    prefix = np.cumsum(ordered)
    # Solve each region of sum(min(1, value/threshold)); this is independent
    # of the Metal bisection and follows CUDA's sorted-prefix construction.
    for index, value in enumerate(ordered):
        denominator = target - (len(ordered) - index - 1)
        if denominator > 0:
            threshold = prefix[index] / denominator
            next_value = ordered[index + 1] if index + 1 < len(ordered) else np.inf
            if value <= threshold <= next_value:
                return threshold
    return 0.0


@pytest.mark.parametrize("rows,regularization", [(1, 0), (1031, .5), (8192, 0), (17003, 2)])
def test_mvs_thresholds_and_reciprocal_probability_reference(probe, rows, regularization):
    derivatives = np.random.default_rng(632).normal(size=rows).astype(np.float32)
    derivatives[::13] = 0
    fraction = float(np.float32(.37))
    result = probe(4, rows, subsample=fraction, mvs_lambda=regularization, derivatives=derivatives)
    magnitudes = np.sqrt(derivatives.astype(np.float64)**2 + regularization)
    thresholds = np.array([_mvs_threshold(magnitudes[start:start + 8192], fraction)
                           for start in range(0, rows, 8192)])
    np.testing.assert_allclose(result["thresholds"], thresholds, rtol=2e-6, atol=1e-7)
    selected = np.repeat(thresholds, 8192)[:rows]
    probabilities = np.divide(magnitudes, selected, out=np.zeros(rows), where=selected > 0)
    probabilities[magnitudes > selected] = 1
    probabilities = np.minimum(probabilities, 1)
    included = (probabilities > np.finfo(np.float32).eps) & (uniforms(rows) < probabilities)
    expected = np.divide(1, probabilities, out=np.zeros(rows), where=included)
    np.testing.assert_allclose(result["values"], expected, rtol=3e-6, atol=1e-7)


def test_mvs_zero_gradients_have_no_nonfinite_weights(probe):
    result = probe(4, derivatives=np.zeros(1031, np.float32))
    np.testing.assert_array_equal(result["thresholds"], 0)
    np.testing.assert_array_equal(result["values"], 0)
    assert np.isfinite(result["gradients"]).all()


def test_mvs_sampling_is_unbiased_and_prioritizes_large_derivatives(probe):
    rows = 131071
    derivatives = np.linspace(.01, 10, rows, dtype=np.float32)
    result = probe(4, rows, derivatives=derivatives, subsample=.4, mvs_lambda=.25)
    weights = result["values"]
    assert abs(np.count_nonzero(weights) / rows - .4) < .008
    assert abs(np.mean(weights) - 1) < .025
    assert abs(np.mean(result["gradients"]) / np.mean(derivatives) - 1) < .025


def test_first_mvs_lambda_reduces_weighted_gradients_on_gpu(probe):
    rows = 1000031
    derivatives = np.random.default_rng(721).normal(size=rows).astype(np.float32)
    derivatives[::11] *= 1e5
    result = probe(7, rows, derivatives=derivatives)
    groups = min((rows + 255) // 256, 4096)
    sums = result["values"][:2 * groups].reshape(-1, 2).sum(axis=0, dtype=np.float64)
    np.testing.assert_allclose(sums[0]**2, np.mean(np.abs(derivatives.astype(np.float64)))**2,
                               rtol=1e-6)
    np.testing.assert_allclose(sums[1], np.mean(derivatives.astype(np.float64)**2), rtol=1e-6)
