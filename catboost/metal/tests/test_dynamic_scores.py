"""GPU candidate masks retain canonical scalar scores and global candidate ids."""

import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


SCORES = ["L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2"]
WINNER = np.dtype([(name, kind) for name, kind in (
    ("index", "u4"), ("feature", "u4"), ("bin", "u4"), ("type", "u4"),
    ("score", "f4"), ("valid", "u4"), ("error", "u4"), ("gain", "f4"),
)])


class KernelParams(ct.Structure):
    _fields_ = ([(name, ct.c_uint32) for name in (
        "rows", "features", "bins", "leaves", "candidates", "split_feature", "split_bin", "split_level")]
        + [(name, ct.c_float) for name in ("bias", "learning_rate", "l2")]
        + [(name, ct.c_uint32) for name in (
            "score_function", "objective", "leaf_iteration", "leaf_iterations", "tile_rows")]
        + [("total_weight", ct.c_float)]
        + [(name, ct.c_uint32) for name in ("histogram_tiles", "partition_tiles", "score_groups")]
        + [("objective_param", ct.c_float), ("leaf_method", ct.c_uint32),
           ("compact", ct.c_uint32), ("score_before_split", ct.c_float)])


assert ct.sizeof(KernelParams) == 96 and KernelParams.score_before_split.offset == 92
assert WINNER.itemsize == 32


@pytest.fixture(scope="module")
def probe():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Dynamic split scoring requires an Apple Silicon Metal GPU")
    root = Path(__file__).resolve().parents[1]
    source = Path(__file__).with_name("dynamic_score_probe.mm")
    headers = [root / "native" / name for name in (
        "metal_kernel_abi.h", "metal_kernels.h", "metal_additional_objective_kernels.h",
        "metal_objective_kernels.h", "metal_streaming_score_kernels.h", "metal_dynamic_score_kernels.h")]
    digest = hashlib.sha256(source.read_bytes() + b"".join(header.read_bytes() for header in headers)).hexdigest()[:20]
    destination = root / ".build" / f"dynamic_score_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                        "-Wno-deprecated-declarations", "-framework", "Foundation", "-framework", "Metal",
                        str(source), "-o", str(destination)], check=True, capture_output=True, text=True)
    library = ct.CDLL(str(destination))
    library.cbm_dynamic_score_probe.argtypes = ([ct.POINTER(KernelParams)] + [ct.c_void_p] * 12
        + [ct.c_uint32, ct.c_uint32, ct.c_void_p, ct.c_char_p, ct.c_uint32])
    library.cbm_dynamic_score_probe.restype = ct.c_int

    def run(problem, active, score="L2", *, dynamic=True, tiles=None, groups=2, score_before=-11.75,
            l2=1.25):
        features = len(problem["offsets"]) - 1
        candidates = len(problem["features"])
        active = np.ascontiguousarray(active, np.uint8)
        assert active.shape == (candidates,)
        if tiles is None:
            tiles = [(2, 3), (4, 7), (0, 2), (3, 4), (7, 7)]
        tile_ranges = np.ascontiguousarray(tiles, np.uint32)
        assert tile_ranges.ndim == 2 and tile_ranges.shape[1] == 2
        arrays = [np.ascontiguousarray(problem[key], dtype) for key, dtype in (
            ("sums", np.float32), ("weights", np.float32), ("leaf_sums", np.float32),
            ("leaf_weights", np.float32), ("features", np.uint32), ("bins", np.uint32),
            ("types", np.uint8), ("noise", np.float32), ("offsets", np.uint32),
            ("penalties", np.float32))]
        arrays += [active, tile_ranges]
        # Keep non-null backing pointers for zero-candidate static references;
        # params.candidates remains zero and the GPU never reads those values.
        arrays = [value if value.size else np.zeros(1, value.dtype) for value in arrays]
        p = KernelParams(rows=1031, features=features, bins=256, leaves=len(problem["leaf_sums"]),
                         candidates=candidates, l2=l2, score_function=SCORES.index(score),
                         score_groups=groups, compact=1, score_before_split=score_before)
        result = np.zeros(len(tile_ranges) + 1, WINNER)
        error = ct.create_string_buffer(4096)
        code = library.cbm_dynamic_score_probe(ct.byref(p), *(value.ctypes.data for value in arrays),
            len(tile_ranges), dynamic, result.ctypes.data, error, len(error))
        assert code == 0, error.value.decode()
        return result

    return run


def _problem(leaves=64, rows=1031, candidates=777):
    rng = np.random.default_rng(581)
    spans = np.array([3, 4, 0, 6, 5, 2, 1], np.uint32)
    offsets = np.concatenate((np.array([0], np.uint32), spans.cumsum(dtype=np.uint32)))
    leaf_ids = rng.integers(0, leaves, rows)
    observation_weights = (2.0 ** rng.integers(-2, 3, rows)).astype(np.float32)
    observation_weights[::17] = 0
    gradient = observation_weights * rng.integers(-32, 33, rows).astype(np.float32) / 8
    sums = np.zeros((leaves, int(offsets[-1])), np.float32)
    weights = np.zeros_like(sums)
    pairs = []
    for feature, span in enumerate(spans):
        values = rng.integers(0, int(span) + 1, rows)
        if feature % 2:
            values[::37] = 255
        valid = values < span
        np.add.at(sums, (leaf_ids[valid], offsets[feature] + values[valid]), gradient[valid])
        np.add.at(weights, (leaf_ids[valid], offsets[feature] + values[valid]), observation_weights[valid])
        if feature % 2 == 0:
            section = slice(int(offsets[feature]), int(offsets[feature + 1]))
            sums[:, section] = sums[:, section].cumsum(axis=1, dtype=np.float32)
            weights[:, section] = weights[:, section].cumsum(axis=1, dtype=np.float32)
        pairs.extend((feature, border) for border in range(int(span)))
    pairs = np.array(pairs, np.uint32)
    selected = pairs[rng.integers(0, len(pairs), candidates)]
    return dict(sums=sums, weights=weights,
                leaf_sums=np.bincount(leaf_ids, weights=gradient, minlength=leaves).astype(np.float32),
                leaf_weights=np.bincount(leaf_ids, weights=observation_weights, minlength=leaves).astype(np.float32),
                features=selected[:, 0].copy(), bins=selected[:, 1].copy(),
                types=(selected[:, 0] % 2).astype(np.uint8), offsets=offsets,
                noise=np.array([-.25, .0625, .03125, -.125, .375, -.75, .5], np.float32),
                penalties=np.array([[1.25, 3], [.5, .25], [1, 1], [2, 2], [1, .5], [.75, 1], [1.5, 4]], np.float32))


def _canonical_subset(probe, problem, active, score, **kwargs):
    """Test-only stable filtering; map the canonical GPU winner back to its id."""
    ids = np.flatnonzero(active)
    filtered = problem | {key: problem[key][ids] for key in ("features", "bins", "types")}
    result = probe(filtered, np.ones(len(ids), np.uint8), score, dynamic=False, **kwargs)
    valid = result["valid"] != 0
    result["index"][valid] = ids[result["index"][valid]]
    return result


def _assert_identical(actual, expected):
    assert actual.tobytes() == expected.tobytes(), (actual, expected)


@pytest.mark.parametrize("score", SCORES)
@pytest.mark.parametrize("mask", ["all", "sparse"])
def test_dynamic_masks_match_canonical_scores_penalties_and_gains(probe, score, mask):
    problem = _problem()
    active = np.ones(len(problem["features"]), np.uint8)
    if mask == "sparse":
        active[np.arange(len(active)) % 3 != 1] = 0
    actual = probe(problem, active, score)
    expected = _canonical_subset(probe, problem, active, score)
    _assert_identical(actual, expected)
    assert actual[-1]["valid"] and active[actual[-1]["index"]]
    assert np.all(actual["error"] == 0)
    # Whole-cache scoring and reordered feature tiles use the same global id.
    whole = probe(problem, active, score, tiles=[(0, 7)])
    _assert_identical(actual[-1:], whole[-1:])


@pytest.mark.parametrize("score", SCORES)
def test_all_disabled_pending_metadata_is_neutral_and_unread(probe, score):
    problem = _problem()
    active = np.zeros(len(problem["features"]), np.uint8)
    problem["features"][:] = np.iinfo(np.uint32).max
    problem["bins"][:] = np.iinfo(np.uint32).max
    problem["types"][:] = 255
    problem["noise"][:] = np.nan
    problem["penalties"][:] = np.nan
    result = probe(problem, active, score)
    assert np.all(result["index"] == np.iinfo(np.uint32).max)
    assert np.all(result["valid"] == 0) and np.all(result["error"] == 0)
    assert np.all(np.isposinf(result["score"])) and np.all(np.isposinf(result["gain"]))


@pytest.mark.parametrize("score", SCORES)
def test_active_ties_keep_original_candidate_id_across_tiles(probe, score):
    problem = _problem()
    problem["sums"][:] = problem["leaf_sums"][:] = 0
    problem["noise"][:] = 0
    problem["penalties"][:] = 1
    active = np.zeros(len(problem["features"]), np.uint8)
    ids = np.array([7, 260, 514, 776])
    active[ids] = [255, 1, 1, 1]  # Every nonzero mask byte enables its candidate.
    problem["features"][ids] = [6, 0, 3, 4]
    problem["bins"][ids] = 0
    problem["types"][ids] = problem["features"][ids] % 2
    options = dict(tiles=[(0, 2), (2, 4), (4, 7)], score_before=0)
    actual = probe(problem, active, score, **options)
    _assert_identical(actual, _canonical_subset(probe, problem, active, score, **options))
    assert actual[0]["index"] == 260 and actual[-1]["index"] == 7
    assert actual[-1]["score"] == actual[-1]["gain"] == 0


@pytest.mark.parametrize("score", SCORES)
def test_masked_nonfinite_candidate_stays_quiet_until_enabled(probe, score):
    problem = _problem()
    bad_feature = 6
    problem["penalties"][bad_feature, 0] = np.nan
    active = (problem["features"] != bad_feature).astype(np.uint8)
    quiet = probe(problem, active, score)
    assert quiet[-1]["valid"] and not quiet[-1]["error"]
    active[np.flatnonzero(problem["features"] == bad_feature)[0]] = 1
    for tiles in ([(6, 7), (0, 6)], [(0, 6), (6, 7)]):
        actual = probe(problem, active, score, tiles=tiles)
        expected = _canonical_subset(probe, problem, active, score, tiles=tiles)
        _assert_identical(actual, expected)
        assert actual[-1]["valid"] and actual[-1]["error"]
        assert actual[-1]["index"] == quiet[-1]["index"]
        bad_tile = actual[0 if tiles[0][0] == 6 else 1]
        assert not bad_tile["valid"] and bad_tile["error"]


@pytest.mark.parametrize("score", ["L2", "Cosine"])
def test_mask_changes_between_depths_without_renumbering_candidates(probe, score):
    problem = _problem(leaves=1024)
    active = np.ones(len(problem["features"]), np.uint8)
    original = probe(problem, active, score)[-1]
    same_split = ((problem["features"] == original["feature"])
                  & (problem["bins"] == original["bin"]))
    active[same_split] = 0
    changed = probe(problem, active, score)
    _assert_identical(changed, _canonical_subset(probe, problem, active, score))
    assert changed[-1]["valid"] and changed[-1]["index"] != original["index"]
    active[same_split] = 1
    restored = probe(problem, active, score)[-1:]
    _assert_identical(restored, np.array([original], WINNER))


@pytest.mark.parametrize("score", ["L2", "Cosine", "NewtonL2", "NewtonCosine"])
def test_one_hot_precision_uses_selected_then_complement_in_every_tile(probe, score):
    # Regression fixture from test_scalar_histogram_precision: routing a
    # selected equality bin right must not reverse CUDA's AddLeaf order. L2
    # rounds its running score after each child; Cosine retains wider terms.
    hg = np.array([[-14.683168411254883, 0.6534653306007385],
                   [11.415841102600098, 2.613861322402954]], np.float32)
    hw = np.array([[29, 3], [57, 12]], np.float32)
    pg = np.array([-14.029703140258789, 14.029702186584473], np.float32)
    pw = np.array([32, 69], np.float32)
    problem = dict(sums=hg, weights=hw, leaf_sums=pg, leaf_weights=pw,
                   features=np.array([0, 1], np.uint32), bins=np.zeros(2, np.uint32),
                   types=np.ones(2, np.uint8), offsets=np.arange(3, dtype=np.uint32),
                   noise=np.zeros(2, np.float32), penalties=np.ones((2, 2), np.float32))
    expected = []
    for candidate in range(2):
        l2_score, numerator, denominator = np.float32(0), 0.0, float(np.float32(1e-10))
        for leaf in range(2):
            selected_g, selected_w = hg[leaf, candidate], hw[leaf, candidate]
            for gradient, weight in ((selected_g, selected_w),
                                     (np.float32(pg[leaf] - selected_g), pw[leaf] - selected_w)):
                gradient, weight = float(gradient), float(weight)
                mean = gradient / (weight + 3)
                l2_score = np.float32(float(l2_score) - gradient * gradient / (weight + 3))
                numerator += gradient * mean
                denominator += weight * mean * mean
        expected.append(np.float32(-numerator / np.sqrt(denominator)) if score.endswith("Cosine") else l2_score)
    options = dict(tiles=[(1, 2), (0, 1)], score_before=0, l2=3)
    actual = probe(problem, np.ones(2, np.uint8), score, **options)
    np.testing.assert_array_equal(actual["score"][:2], expected[::-1])
    np.testing.assert_array_equal(actual["gain"][:2], expected[::-1])
    assert actual[-1]["index"] == np.argmin(expected)
    assert actual[-1]["type"] == 1
    masked = probe(problem, np.array([1, 0], np.uint8), score, **options)
    assert not masked[0]["valid"] and not masked[0]["error"]
    assert masked[-1]["score"] == expected[0] and masked[-1]["index"] == 0
