"""Exercise the actual Metal weak-noise kernel against independent scalar math.

The integer oracle follows the documented Metal per-item seed expansion and
CUDA MWC recurrence; it does not claim NVIDIA launch-thread seed equivalence.
No CatBoost fitting or production Python RNG helper is used.
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


class LangevinWeakParams(ct.Structure):
    _fields_ = [("random", BootstrapParams)] + [(name, ct.c_uint32) for name in (
        "offset", "stride", "filter_bootstrap", "reserved"
    )]


MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
UNIT32 = np.float32(2.0**-32)
LAST_UNIFORM = np.nextafter(np.float32(1), np.float32(0))


def _mix_seed(value):
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def _item_seed(item, seed, iteration, stream):
    state = _mix_seed(seed ^ _mix_seed(iteration ^ 0xD1B54A32D192ED03)
                      ^ _mix_seed(((stream << 32) | item) ^ 0x94D049BB133111EB))
    if not state & MASK32:
        state |= 1
    if not state >> 32:
        state |= 1 << 32
    return state


def _next_word(state):
    high, low = state >> 32, state & MASK32
    high = (36969 * (high & 0xFFFF) + (high >> 16)) & MASK32
    low = (18000 * (low & 0xFFFF) + (low >> 16)) & MASK32
    return (high << 32) | low, ((high << 16) + low) & MASK32


def _words_and_normals(rows, *, seed, iteration, stream):
    words = np.empty((rows, 2), np.uint32)
    normals = np.empty(rows, np.float32)
    for row in range(rows):
        state = _item_seed(row, seed, iteration, stream)
        # There are no BootstrapNormalForItem/score-noise warm-up draws here.
        state, first = _next_word(state)
        _, second = _next_word(state)
        words[row] = first, second
        first_uniform = max(min(np.float32(first) * UNIT32, LAST_UNIFORM), UNIT32)
        second_uniform = min(np.float32(second) * UNIT32, LAST_UNIFORM)
        radius = np.float32(-2) * np.float32(math.log(float(first_uniform)))
        angle = np.float32(2 * math.pi) * second_uniform
        normals[row] = np.float32(math.sqrt(float(radius))) * np.float32(math.cos(float(angle)))
    return words, normals


def _coefficient(temperature, learning_rate):
    # Both public parameters enter the native source helper as float32. The
    # source expression evaluates its divisions and sqrt in double, then the
    # weak GPU kernel receives a float coefficient.
    temperature, learning_rate = float(np.float32(temperature)), float(np.float32(learning_rate))
    return np.float32(0 if temperature == 0 else math.sqrt(2.0 / learning_rate / temperature))


@pytest.fixture(scope="module")
def probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Actual Apple Silicon Metal GPU required")
    source = Path(__file__).with_name("langevin_weak_probe.mm")
    root = source.parent.parent
    inputs = [source, root / "native/metal_bootstrap_kernels.h", root / "native/metal_langevin_kernels.h"]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in inputs)).hexdigest()[:20]
    destination = root / ".build" / f"langevin_weak_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        result = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(destination)],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    library = ct.CDLL(str(destination))
    library.cbm_langevin_weak_probe.argtypes = [ct.POINTER(LangevinWeakParams)] + [ct.c_uint32] * 3 + [ct.c_void_p] * 5 + [ct.c_uint32]
    library.cbm_langevin_weak_probe.restype = ct.c_int

    def run(values, multipliers, *, seed=0xFEDCBA9876543210, iteration=17, stream=0x4C000001,
            temperature=1000, learning_rate=0.03, offset=0, stride=1, filter_bootstrap=False,
            launch_count=None, threadgroup_size=256):
        values = np.ascontiguousarray(values, dtype=np.float32)
        multipliers = np.ascontiguousarray(multipliers, dtype=np.float32)
        assert values.ndim == multipliers.ndim == 1
        rows = len(multipliers)
        coefficient = _coefficient(temperature, learning_rate)
        random = BootstrapParams(rows, 0, seed & MASK32, seed >> 32, iteration, stream,
                                 0, 0, 0, 1, 0, coefficient)
        params = LangevinWeakParams(random, offset, stride, filter_bootstrap, 0)
        output = np.empty_like(values)
        words = np.empty((rows, 2), np.uint32)
        error = ct.create_string_buffer(4096)
        code = library.cbm_langevin_weak_probe(ct.byref(params), len(values),
            rows if launch_count is None else launch_count, threadgroup_size, values.ctypes.data,
            multipliers.ctypes.data, output.ctypes.data, words.ctypes.data, error, len(error))
        assert code == 0, error.value.decode()
        return output, words

    return run


def _assert_reference(probe, values, multipliers, *, seed=0xFEDCBA9876543210,
                      iteration=17, stream=0x4C000001, temperature=1000, learning_rate=0.03,
                      offset=0, stride=1, filter_bootstrap=False, **launch):
    values, multipliers = np.asarray(values, np.float32), np.asarray(multipliers, np.float32)
    output, words = probe(values, multipliers, seed=seed, iteration=iteration, stream=stream,
        temperature=temperature, learning_rate=learning_rate, offset=offset, stride=stride,
        filter_bootstrap=filter_bootstrap, **launch)
    expected_words, normal = _words_and_normals(len(multipliers), seed=seed, iteration=iteration, stream=stream)
    np.testing.assert_array_equal(words, expected_words)
    coefficient = _coefficient(temperature, learning_rate)
    expected = values.copy()
    indices = offset + np.arange(len(multipliers)) * stride
    eligible = multipliers != 0 if filter_bootstrap else np.ones(len(multipliers), bool)
    selected = indices[eligible]
    expected[selected] = expected[selected] + coefficient * normal[eligible]
    np.testing.assert_allclose(output[selected], expected[selected], rtol=3e-5,
                               atol=max(float(coefficient) * 3e-6, 1e-7))
    untouched = np.ones(len(values), bool)
    untouched[selected] = False
    np.testing.assert_array_equal(output[untouched].view(np.uint32), values[untouched].view(np.uint32))
    return output, words


@pytest.mark.parametrize("rows", [1, 127, 256, 257, 1031, 8193])
@pytest.mark.parametrize("temperature,learning_rate", [(10, 0.03125), (10000, 0.03), (250, 0.2)])
def test_additive_coefficient_and_exact_integer_random_reference(probe, rows, temperature, learning_rate):
    _assert_reference(probe, np.linspace(-0.75, 1.25, rows, dtype=np.float32), np.ones(rows),
                      temperature=temperature, learning_rate=learning_rate)


@pytest.mark.parametrize("offset", [0, 5])
def test_ordered_stride_two_only_changes_gradient_component(probe, offset):
    rows = 257
    values = np.linspace(-4, 9, offset + 2 * rows + 7, dtype=np.float32)
    values[offset:offset + rows * 2:2] = np.linspace(-1, 1, rows)
    output, _ = _assert_reference(probe, values, np.ones(rows), offset=offset, stride=2,
                                 launch_count=rows + 513, threadgroup_size=64)
    np.testing.assert_array_equal(output[offset + 1:offset + rows * 2:2],
                                  values[offset + 1:offset + rows * 2:2])


def test_call_domains_iterations_and_dispatch_shapes(probe):
    values, multipliers = np.zeros(513, np.float32), np.ones(513, np.float32)
    first, first_words = _assert_reference(probe, values, multipliers, stream=0x4C000001)
    repeat, repeat_words = probe(values, multipliers, stream=0x4C000001, threadgroup_size=32)
    np.testing.assert_array_equal(first, repeat)
    np.testing.assert_array_equal(first_words, repeat_words)
    next_call, next_words = _assert_reference(probe, values, multipliers, stream=0x4C000002)
    next_iteration, iteration_words = _assert_reference(probe, values, multipliers, iteration=18)
    assert np.any(first_words != next_words)
    assert np.any(first_words != iteration_words)
    assert not np.array_equal(first, next_call)
    assert not np.array_equal(first, next_iteration)
    prefix, prefix_words = probe(values[:17], multipliers[:17], launch_count=4096)
    np.testing.assert_array_equal(prefix, first[:17])
    np.testing.assert_array_equal(prefix_words, first_words[:17])


@pytest.mark.parametrize("filter_bootstrap", [False, True])
@pytest.mark.parametrize("stride", [1, 2])
def test_zero_temperature_preserves_every_float_bit(probe, filter_bootstrap, stride):
    # NaN payload and negative zero make this stronger than arithmetic equality:
    # a zero-scale multiply/add would not preserve all these original bits.
    bits = np.array([0, 0x80000000, 0x7FC12345, 0x3F800001, 0xBF800000, 0x7F800000], np.uint32)
    values = np.tile(bits, 9).view(np.float32)
    rows = (len(values) - 3) // stride
    multipliers = np.resize(np.array([0, 2, 1, 0], np.float32), rows)
    result, _ = probe(values, multipliers, offset=3, stride=stride,
                      temperature=0, filter_bootstrap=filter_bootstrap)
    np.testing.assert_array_equal(result.view(np.uint32), values.view(np.uint32))


@pytest.mark.parametrize("filter_bootstrap", [False, True])
def test_mask_uses_only_bootstrap_multiplier_not_original_weight(probe, filter_bootstrap):
    multipliers = np.resize(np.array([0, 1, 4, 1e-20, 0, 2], np.float32), 257)
    original_weights = np.resize(np.array([0, 0, 0, 3, 9, 0], np.float32), 257)
    values = original_weights * np.float32(0.125)
    output, _ = _assert_reference(probe, values, multipliers, filter_bootstrap=filter_bootstrap)
    zero_weight_kept = (original_weights == 0) & ((multipliers != 0) if filter_bootstrap else True)
    assert np.all(output[zero_weight_kept] != values[zero_weight_kept])
    if filter_bootstrap:
        np.testing.assert_array_equal(output[multipliers == 0], values[multipliers == 0])
    else:
        assert np.all(output[multipliers == 0] != values[multipliers == 0])


def test_noise_is_added_once_and_not_scaled_by_positive_weights(probe):
    rows = 1031
    zeros = np.zeros(rows, np.float32)
    baseline_noise, _ = probe(zeros, np.ones(rows), filter_bootstrap=True)
    varied_multipliers = np.resize(np.array([0.125, 2, 100, 1e-10], np.float32), rows)
    unscaled_noise, _ = probe(zeros, varied_multipliers, filter_bootstrap=True)
    np.testing.assert_array_equal(unscaled_noise, baseline_noise)
    original_weighted_gradient = np.linspace(-1, 1, rows, dtype=np.float32)
    result, _ = probe(original_weighted_gradient, varied_multipliers, filter_bootstrap=True)
    np.testing.assert_allclose(result, original_weighted_gradient + baseline_noise, rtol=2e-7, atol=2e-7)


def test_sequential_calls_use_distinct_noise_and_one_addition_per_call(probe):
    rows = 257
    values, multipliers = np.linspace(-0.25, 0.25, rows, dtype=np.float32), np.ones(rows)
    first, _ = _assert_reference(probe, values, multipliers, stream=0x4C000001)
    second, _ = _assert_reference(probe, first, multipliers, stream=0x4C000002)
    repeated_first, _ = probe(first, multipliers, stream=0x4C000001)
    assert not np.array_equal(second, repeated_first)


def test_all_filtered_zero_draw_and_unfiltered_zero_multipliers(probe):
    values = np.linspace(-2, 2, 257, dtype=np.float32)
    filtered, _ = probe(values, np.zeros(257), filter_bootstrap=True)
    np.testing.assert_array_equal(filtered, values)
    unfiltered, _ = _assert_reference(probe, values, np.zeros(257), filter_bootstrap=False)
    assert np.all(unfiltered != values)
