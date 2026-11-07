"""Numeric Ordered fold arithmetic on Metal, independently checked from CUDA.

No CPU CatBoost training is called. CUDA source references are in ORDERED_PORT.md.
"""
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


class OrderedParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "folds", "leaves", "candidates", "packed_rows", "test_only", "score_function")]
    _fields_ += [("l2", ct.c_float), ("normalize", ct.c_uint32),
                ("score_before", ct.c_float), ("learning_rate", ct.c_float)]


@pytest.fixture(scope="module")
def ordered_probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    source = Path(__file__).with_name("ordered_probe.mm")
    root = source.parent.parent
    header = root / "native/metal_ordered_kernels.h"
    digest = hashlib.sha256(source.read_bytes() + header.read_bytes()).hexdigest()[:16]
    destination = root / ".build" / f"ordered_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-framework", "Foundation", "-framework", "Metal", str(source),
                        "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_ordered_folds.argtypes = [ct.c_uint32, ct.c_float, ct.c_uint32, ct.c_void_p,
                                         ct.c_uint32, ct.c_void_p, ct.c_void_p, ct.c_uint32]
    library.cbm_ordered_folds.restype = ct.c_int
    library.cbm_ordered_probe.argtypes = [ct.POINTER(OrderedParams)] + [ct.c_void_p] * 19 + [ct.c_uint32]
    library.cbm_ordered_probe.restype = ct.c_int

    class Probe:
        @staticmethod
        def folds(rows, growth=2, minimum=100):
            output = np.zeros((4096, 4), np.uint32)
            count, error = ct.c_uint32(), ct.create_string_buffer(4096)
            code = library.cbm_ordered_folds(rows, growth, minimum, output.ctypes.data, 4096,
                                            ct.byref(count), error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return output[:count.value].copy()

        @staticmethod
        def run(bins, targets, weights, cursors, folds, *, permutation=None,
                leaf_ids=None, leaves=None, candidates=None, split_types=None,
                multipliers=None, test_only=True, score_function=0, l2=3,
                normalize=False, score_before=0, learning_rate=.1, feature_options=None):
            bins = np.ascontiguousarray(bins, np.uint8)
            rows, features = bins.shape
            targets, weights, cursors = (np.ascontiguousarray(x, np.float32) for x in (targets, weights, cursors))
            folds = np.ascontiguousarray(folds, np.uint32)
            permutation = np.ascontiguousarray(np.arange(rows) if permutation is None else permutation, np.uint32)
            leaf_ids = np.ascontiguousarray(np.zeros(rows) if leaf_ids is None else leaf_ids, np.uint32)
            leaves = int(leaf_ids.max()) + 1 if leaves is None else leaves
            candidates = np.ascontiguousarray([[0, 0]] if candidates is None else candidates, np.uint32)
            split_types = np.ascontiguousarray(np.zeros(len(candidates)) if split_types is None else split_types, np.uint32)
            multipliers = np.ascontiguousarray(np.ones(len(cursors)) if multipliers is None else multipliers, np.float32)
            feature_options = np.ascontiguousarray(np.tile([1, 1, 0, 0], (features, 1))
                if feature_options is None else feature_options, np.float32)
            assert targets.shape == weights.shape == permutation.shape == leaf_ids.shape == (rows,)
            assert folds.shape[1] == 4 and len(split_types) == len(candidates)
            assert multipliers.shape == cursors.shape and feature_options.shape == (features, 4)
            params = OrderedParams(rows, features, len(folds), leaves, len(candidates), len(cursors),
                test_only, score_function, l2, normalize, score_before, learning_rate)
            outputs = [np.zeros((len(cursors), 2), np.float32), np.zeros((len(cursors), 2), np.float32),
                np.zeros((len(candidates), leaves, len(folds), 2, 4), np.float32),
                np.zeros((len(candidates), 2), np.float32), np.zeros((len(folds), 2), np.float32),
                np.zeros((len(folds), leaves), np.float32), np.zeros_like(cursors)]
            error = ct.create_string_buffer(4096)
            inputs = [bins, permutation, leaf_ids, candidates, split_types, folds, targets, weights,
                      cursors, multipliers, feature_options]
            code = library.cbm_ordered_probe(ct.byref(params),
                *[x.ctypes.data for x in inputs + outputs], error, len(error))
            if code:
                raise ValueError(error.value.decode())
            return dict(zip(("derivatives", "sampled", "statistics", "scores", "quality", "leaf_values", "cursors"), outputs))
    return Probe


def dataset(probe, rows=1027, seed=1337):
    rng = np.random.default_rng(seed)
    folds = probe.folds(rows)
    return dict(bins=rng.integers(0, 9, (rows, 3), dtype=np.uint8),
        targets=rng.normal(size=rows).astype(np.float32),
        weights=rng.choice([0, .1, 1, 3], size=rows).astype(np.float32),
        cursors=rng.normal(size=int(folds[-1, 2] + folds[-1, 1])).astype(np.float32), folds=folds,
        permutation=rng.permutation(rows), leaf_ids=rng.integers(0, 4, rows), leaves=5,
        candidates=np.array([[0, 1], [0, 3], [1, 2], [2, 4]]), split_types=[0, 0, 1, 0],
        multipliers=rng.choice([0, .5, 1, 4], size=int(folds[-1, 2] + folds[-1, 1])).astype(np.float32))


def reference(data, *, test_only=True, score_function=0, l2=3, normalize=False,
              score_before=0, learning_rate=.1, feature_options=None):
    """Double-precision equations evaluated directly from selected row slices."""
    bins = np.asarray(data["bins"])
    y, w, cursor = (np.asarray(data[k], np.float32).astype(np.float64) for k in ("targets", "weights", "cursors"))
    permutation = np.asarray(data["permutation"])
    leaves, folds = data["leaves"], data["folds"]
    types, candidates, leaf_ids = np.asarray(data["split_types"]), np.asarray(data["candidates"]), data["leaf_ids"]
    boot = np.asarray(data["multipliers"], np.float32).astype(np.float64)
    l2, score_before, learning_rate = (float(np.float32(x)) for x in (l2, score_before, learning_rate))
    options = np.tile([1, 1, 0, 0], (bins.shape[1], 1)) if feature_options is None else np.asarray(feature_options, np.float32)
    derivatives, sampled = np.zeros((len(cursor), 2)), np.zeros((len(cursor), 2))
    stats = np.zeros((len(candidates), leaves, len(folds), 2, 4))
    quality = np.zeros((len(folds), 2))
    values, updated = np.zeros((len(folds), leaves)), cursor.copy()
    for f, (prefix, end, offset, _) in enumerate(folds):
        prefix, end, offset = map(int, (prefix, end, offset))
        rows = permutation[:end]
        # Native objective arithmetic receives float32: subtract then multiply.
        gradient = ((y[rows].astype(np.float32) - cursor[offset:offset + end].astype(np.float32))
                    * w[rows].astype(np.float32)).astype(np.float64)
        derivatives[offset:offset + end] = np.column_stack((gradient, w[rows]))
        multipliers = boot[offset:offset + end].copy()
        if test_only:
            multipliers[:prefix] = 1
        sampled[offset:offset + end] = (derivatives[offset:offset + end].astype(np.float32)
                                       * multipliers.astype(np.float32)[:, None]).astype(np.float64)
        tail_g, tail_w = gradient[prefix:], w[rows[prefix:]]
        weak = np.divide(tail_g, tail_w + np.float32(1e-15), out=np.zeros_like(tail_g),
                         where=np.abs(tail_g) >= np.float32(1e-15))
        quality[f] = [np.dot(weak * weak, tail_w), end - prefix]
        for leaf in range(leaves):
            estimate_rows = leaf_ids[rows[:prefix]] == leaf
            estimate_mass = w[rows[:prefix]][estimate_rows].sum()
            denominator = estimate_mass + l2 * (w[rows[:prefix]].sum() if normalize else 1)
            if estimate_mass > 1e-20 and denominator > 0:
                values[f, leaf] = gradient[:prefix][estimate_rows].sum() / denominator
        updated[offset:offset + end] += learning_rate * values[f, leaf_ids[rows]]
        for c, (feature, border) in enumerate(candidates):
            side = bins[rows, feature] == border if types[c] else bins[rows, feature] > border
            for leaf in range(leaves):
                for side_id in (0, 1):
                    mask = (leaf_ids[rows] == leaf) & (side == side_id)
                    prefix_mask, tail_mask = mask.copy(), mask.copy()
                    prefix_mask[prefix:] = False
                    tail_mask[:prefix] = False
                    ds = sampled[offset:offset + end]
                    stats[c, leaf, f, side_id] = [ds[prefix_mask, 1].sum(), ds[prefix_mask, 0].sum(),
                                                ds[tail_mask, 1].sum(), ds[tail_mask, 0].sum()]
    scores = np.zeros((len(candidates), 2))
    for c, (feature, _) in enumerate(candidates):
        s = stats[c]
        denominator = s[..., 0] + ((l2 * s[..., 0] if normalize else l2) if not score_function else 1e-15)
        mu = np.divide(s[..., 1], denominator, out=np.zeros_like(denominator), where=s[..., 0] > 0)
        if score_function:
            # CUDA accumulates the strict >2 gating mass in float32. A
            # double sum of stored 0.1f values can spuriously cross 2.
            mass = s[..., 2].sum(axis=1).astype(np.float32).astype(np.float64)
            tail_error = (-2 * mu * s[..., 3] + s[..., 2] * mu * mu).sum(axis=1)
            score = np.where(mass > 2, tail_error * (1 + 2 * np.log1p(mass)), 0).sum()
        else:
            norm = np.float32(1e-20) + (s[..., 2] * mu * mu).sum()
            score = -(s[..., 3] * mu).sum() / np.sqrt(norm) if norm > np.float32(1e-15) else np.finfo(np.float32).max
        score = score * options[feature, 0] + (options[feature, 2] if not score_function else 0)
        scores[c] = [score, (score - score_before) * options[feature, 1]]
    return dict(derivatives=derivatives, sampled=sampled, statistics=stats, scores=scores,
                quality=quality, leaf_values=values, cursors=updated)


@pytest.mark.parametrize("rows,expected", [
    (4, [(2, 4)]), (9, [(2, 5), (5, 9)]),
    (500, [(11, 23), (23, 47), (47, 95), (95, 191), (191, 383), (383, 500)]),
    (10000, [(101, 203), (203, 407), (407, 815), (815, 1631), (1631, 3263), (3263, 6527), (6527, 10000)])])
def test_numeric_folds_match_cuda_growth(ordered_probe, rows, expected):
    folds = ordered_probe.folds(rows)
    np.testing.assert_array_equal(folds[:, :2], expected)
    np.testing.assert_array_equal(folds[:, 2], np.r_[0, np.cumsum(folds[:-1, 1])])


def test_fold_growth_cannot_stall_for_small_prefix(ordered_probe):
    folds = ordered_probe.folds(9, growth=1.1)
    np.testing.assert_array_equal(folds[:, :2], list(zip(range(2, 9), range(3, 10))))


def test_large_dataset_minimum_caps_fold_growth(ordered_probe):
    folds = ordered_probe.folds(1 << 24, minimum=1)
    assert folds[0, 0] == 65 and folds[-1, 1] == 1 << 24 and len(folds) == 18


@pytest.mark.parametrize("rows,growth,minimum", [(3, 2, 100), (10, 1, 100), (10, np.nan, 100), (10, 2, 0)])
def test_invalid_fold_configuration(ordered_probe, rows, growth, minimum):
    with pytest.raises(ValueError, match="configuration"):
        ordered_probe.folds(rows, growth, minimum)


@pytest.mark.parametrize("test_only", [False, True])
@pytest.mark.parametrize("score_function", [0, 1])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("rows", [33, 1027])
def test_fold_statistics_scores_and_cursor_update(ordered_probe, test_only, score_function, normalize, rows):
    data = dataset(ordered_probe, rows)
    config = dict(test_only=test_only, score_function=score_function, normalize=normalize,
                  l2=2.3, score_before=-.7, learning_rate=.08,
                  feature_options=[[.9, 1.1, .03, 0], [1, .7, -.07, 0], [1.3, .6, .1, 0]])
    result = ordered_probe.run(**data, **config)
    expected = reference(data, **config)
    for key in result:
        np.testing.assert_allclose(result[key], expected[key], rtol=7e-6, atol=3e-5, err_msg=key)


def test_quality_target_does_not_leak_into_prefix_leaf_estimation(ordered_probe):
    data = dataset(ordered_probe, 33)
    data["folds"] = data["folds"][-1:].copy()
    data["cursors"] = data["cursors"][-33:]
    data["multipliers"] = data["multipliers"][-33:]
    data["folds"][0, 2] = 0
    data["weights"][:] = 1
    before = ordered_probe.run(**data)
    tail = data["permutation"][int(data["folds"][0, 0]):]
    data["targets"][tail] += 1000
    after = ordered_probe.run(**data)
    np.testing.assert_array_equal(before["leaf_values"], after["leaf_values"])
    np.testing.assert_array_equal(before["cursors"], after["cursors"])
    assert not np.array_equal(before["quality"], after["quality"])


def test_bootstrap_changes_scores_but_not_exported_leaf_estimation_or_noise_variance(ordered_probe):
    data = dataset(ordered_probe)
    before = ordered_probe.run(**data)
    data["multipliers"] = np.ones_like(data["multipliers"])
    after = ordered_probe.run(**data)
    for key in ("leaf_values", "cursors", "quality", "derivatives"):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before["scores"], after["scores"])


def test_same_document_has_independent_residual_per_fold(ordered_probe):
    data = dataset(ordered_probe, 33)
    result = ordered_probe.run(**data)
    positions = data["folds"][:, 2].astype(int)
    original = data["permutation"][0]
    expected = data["weights"][original] * (data["targets"][original] - data["cursors"][positions])
    np.testing.assert_array_equal(result["derivatives"][positions, 0], expected)
    assert np.unique(data["cursors"][positions]).size == len(positions)


def test_full_estimation_task_uses_all_rows_with_its_own_cursor(ordered_probe):
    data = dataset(ordered_probe, 33)
    packed = len(data["cursors"])
    data["folds"] = np.vstack([data["folds"], [33, 33, packed, 0]])
    estimation_cursor = np.linspace(-.5, .5, 33).astype(np.float32)
    data["cursors"] = np.r_[data["cursors"], estimation_cursor]
    data["multipliers"] = np.r_[data["multipliers"], np.ones(33)]
    result = ordered_probe.run(**data, normalize=True)
    expected = reference(data, normalize=True)
    np.testing.assert_allclose(result["leaf_values"][-1], expected["leaf_values"][-1], rtol=2e-6, atol=2e-7)
    assert not np.allclose(result["leaf_values"][-1], result["leaf_values"][-2])
    np.testing.assert_array_equal(result["quality"][-1], [0, 0])


def test_zero_weight_folds_and_empty_leaves_are_safe(ordered_probe):
    data = dataset(ordered_probe, 33)
    data["weights"][:] = 0
    for function in (0, 1):
        result = ordered_probe.run(**data, score_function=function, l2=0, normalize=True)
        np.testing.assert_array_equal(result["leaf_values"], 0)
        np.testing.assert_array_equal(result["cursors"], data["cursors"])
        np.testing.assert_array_equal(result["scores"][:, 0], np.finfo(np.float32).max if function == 0 else 0)


def test_solar_ignores_l2_normalization_and_feature_noise(ordered_probe):
    data = dataset(ordered_probe)
    before = ordered_probe.run(**data, score_function=1, l2=0)
    after = ordered_probe.run(**data, score_function=1, l2=10000, normalize=True,
                              feature_options=np.tile([1, 1, 100, 0], (3, 1)))
    np.testing.assert_array_equal(before["scores"], after["scores"])


@pytest.mark.parametrize("field", ["permutation", "folds", "multipliers", "candidates"])
def test_invalid_runtime_input_is_rejected_before_dispatch(ordered_probe, field):
    data = dataset(ordered_probe, 33)
    if field == "permutation":
        data[field][0] = data[field][1]
    elif field == "folds":
        data[field][1, 2] += 1
    elif field == "multipliers":
        data[field][0] = -1
    else:
        data[field][0, 0] = 3
    with pytest.raises(ValueError, match="Invalid Ordered"):
        ordered_probe.run(**data)
