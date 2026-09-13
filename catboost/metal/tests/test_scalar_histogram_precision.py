"""Expose actual scalar Metal histogram stages without training a CatBoost model.

The compiled dylib embeds all MSL strings, so retaining its path preserves a
baseline even while native headers change. ``--compile-only`` does no GPU work.
Input NPZ keys: bins [features, rows], gradients/weights [rows], row_indices
[rows], partition_offsets [3], feature_types [features], and optional bin_count.
"""

import argparse
import ctypes as ct
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np
import pytest


_STAGES = {
    "root_raw": (0, 1), "root_prefix": (1, 1),
    "smaller_raw": (2, 1), "smaller_prefix": (3, 1),
    "children_reused": (4, 2), "children_rebuild_raw": (6, 2),
    "children_rebuilt": (8, 2),
}


def build_probe():
    """Compile host Objective-C++ only; the resulting library retains its headers."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Requires Apple Silicon macOS")
    source = Path(__file__).with_name("scalar_histogram_probe.mm")
    root = source.parent.parent
    headers = [root / "native" / name for name in (
        "metal_kernel_abi.h", "metal_kernels.h", "metal_additional_objective_kernels.h",
        "metal_objective_kernels.h", "metal_histogram_kernels.h",
        "metal_histogram_reuse_kernels.h", "metal_incremental_partition_kernels.h",
    )]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in [source, *headers])).hexdigest()[:16]
    destination = root / ".build" / f"scalar_histogram_probe_{digest}.dylib"
    destination.parent.mkdir(exist_ok=True)
    if not destination.exists():
        subprocess.run([
            "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
            "-framework", "Foundation", "-framework", "Metal", str(source), "-o", str(destination),
        ], check=True, capture_output=True, text=True)
    return destination


def run_probe(bins, gradients, weights, row_indices, partition_offsets, feature_types,
              *, bin_count=None, tiles=1, rebuild=True, library_path=None):
    """Dispatch production kernels and return G/W snapshots plus parent statistics."""
    original_bins = np.asarray(bins)
    if original_bins.ndim != 2 or min(original_bins.shape) < 1:
        raise ValueError("bins must have shape [features, rows] with positive dimensions")
    features, rows = original_bins.shape
    if not np.issubdtype(original_bins.dtype, np.integer) or np.any(original_bins < 0) or np.any(original_bins > 255):
        raise ValueError("bins must be integers in [0, 255]")
    bin_count = int(original_bins.max()) + 1 if bin_count is None else int(bin_count)
    if not 1 <= bin_count <= 256 or np.any(original_bins >= bin_count):
        raise ValueError("bin_count must contain every bin and be at most 256")
    if not 1 <= tiles <= 256:
        raise ValueError("tiles must be in [1, 256]")
    indices = np.asarray(row_indices)
    offsets = np.asarray(partition_offsets)
    types = np.asarray(feature_types)
    if indices.shape != (rows,) or not np.issubdtype(indices.dtype, np.integer) or not np.array_equal(np.sort(indices), np.arange(rows)):
        raise ValueError("row_indices must be a permutation of the rows")
    if offsets.shape != (3,) or not np.issubdtype(offsets.dtype, np.integer) or offsets[0] != 0 or offsets[2] != rows or not 0 <= offsets[1] <= rows:
        raise ValueError("partition_offsets must be [0, left_count, rows]")
    if types.shape != (features,) or not np.all((types == 0) | (types == 1)):
        raise ValueError("feature_types must contain one numeric=0/onehot=1 flag per feature")
    g = np.ascontiguousarray(gradients, np.float32)
    w = np.ascontiguousarray(weights, np.float32)
    if g.shape != (rows,) or w.shape != (rows,) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("gradients and nonnegative weights must be finite row vectors")
    arrays = [np.ascontiguousarray(original_bins, np.uint8), g, w,
              np.ascontiguousarray(indices, np.uint32), np.ascontiguousarray(offsets, np.uint32),
              np.ascontiguousarray(types, np.uint8)]
    gradient_stages = np.zeros((10, features, bin_count), np.float32)
    weight_stages = np.zeros_like(gradient_stages)
    partition_g = np.zeros(3, np.float32)
    partition_w = np.zeros(3, np.float32)
    arrays.extend([gradient_stages, weight_stages, partition_g, partition_w])
    library = ct.CDLL(str(build_probe() if library_path is None else library_path))
    function = library.cbm_scalar_histogram_probe
    function.argtypes = [ct.c_uint32] * 5 + [ct.c_void_p] * 11 + [ct.c_uint32]
    function.restype = ct.c_int
    error = ct.create_string_buffer(4096)
    code = function(rows, features, bin_count, tiles, rebuild,
                    *(a.ctypes.data for a in arrays), error, len(error))
    if code:
        raise RuntimeError(error.value.decode())
    result = {"partition_gradients": partition_g, "partition_weights": partition_w,
              "smaller_child": int(offsets[1] >= rows - offsets[1])}
    for name, (start, count) in _STAGES.items():
        if not rebuild and name in ("children_rebuild_raw", "children_rebuilt"):
            continue
        selection = start if count == 1 else slice(start, start + count)
        result[name] = {"gradients": gradient_stages[selection], "weights": weight_stages[selection]}
    return result


def reference_histograms(bins, gradients, weights, row_indices, partition_offsets,
                         feature_types, *, bin_count):
    """Float64 mathematical reference; deliberately independent of GPU reductions."""
    bins = np.asarray(bins)
    features, rows = bins.shape
    g, w = (np.asarray(a, np.float32).astype(np.float64) for a in (gradients, weights))
    indices = np.asarray(row_indices)
    offsets = np.asarray(partition_offsets)
    numeric = np.asarray(feature_types) == 0
    selections = [np.arange(rows), indices[:offsets[1]], indices[offsets[1]:]]
    raw_g, raw_w = np.zeros((3, features, bin_count)), np.zeros((3, features, bin_count))
    for leaf, selected in enumerate(selections):
        for feature in range(features):
            raw_g[leaf, feature] = np.bincount(bins[feature, selected], weights=g[selected], minlength=bin_count)
            raw_w[leaf, feature] = np.bincount(bins[feature, selected], weights=w[selected], minlength=bin_count)
    prefix_g, prefix_w = raw_g.copy(), raw_w.copy()
    prefix_g[:, numeric] = np.cumsum(raw_g[:, numeric], axis=-1)
    prefix_w[:, numeric] = np.cumsum(raw_w[:, numeric], axis=-1)
    smaller = int(offsets[1] >= rows - offsets[1])
    return {
        "root_raw": {"gradients": raw_g[0], "weights": raw_w[0]},
        "root_prefix": {"gradients": prefix_g[0], "weights": prefix_w[0]},
        "smaller_raw": {"gradients": raw_g[1 + smaller], "weights": raw_w[1 + smaller]},
        "smaller_prefix": {"gradients": prefix_g[1 + smaller], "weights": prefix_w[1 + smaller]},
        "children_reused": {"gradients": prefix_g[1:], "weights": prefix_w[1:]},
        "children_rebuild_raw": {"gradients": raw_g[1:], "weights": raw_w[1:]},
        "children_rebuilt": {"gradients": prefix_g[1:], "weights": prefix_w[1:]},
        "partition_gradients": np.array([g[s].sum() for s in selections]),
        "partition_weights": np.array([w[s].sum() for s in selections]),
        "smaller_child": smaller,
    }


def score_histograms(histogram_gradients, histogram_weights, partition_gradients,
                     partition_weights, candidate_features, candidate_bins, candidate_types,
                     *, rows, l2=3, score_function=1, feature_penalties=None, library_path=None):
    """Return each candidate's score from production FindSplitWinners on Metal."""
    hg, hw = (np.ascontiguousarray(v, np.float32) for v in (histogram_gradients, histogram_weights))
    if hg.ndim == 2:
        hg, hw = hg[None], hw[None]
    if hg.ndim != 3 or hw.shape != hg.shape or min(hg.shape) < 1:
        raise ValueError("histograms must have matching [leaves, features, bins] shapes")
    leaves, features, bins = hg.shape
    pg, pw = (np.ascontiguousarray(v, np.float32) for v in (partition_gradients, partition_weights))
    cf, cb, types = (np.asarray(v) for v in (candidate_features, candidate_bins, candidate_types))
    if pg.shape != (leaves,) or pw.shape != (leaves,):
        raise ValueError("partition statistics must have one entry per leaf")
    if cf.ndim != 1 or not cf.size or cb.shape != cf.shape or types.shape != cf.shape:
        raise ValueError("candidate arrays must have matching nonempty vector shapes")
    if not np.issubdtype(cf.dtype, np.integer) or not np.issubdtype(cb.dtype, np.integer) or np.any(cf < 0) or np.any(cf >= features) or np.any(cb < 0) or np.any(cb >= bins) or np.any((types != 0) & (types != 1)):
        raise ValueError("invalid candidate index or type")
    penalties = np.ones((features, 2), np.float32) if feature_penalties is None else np.ascontiguousarray(feature_penalties, np.float32)
    if penalties.shape != (features, 2):
        raise ValueError("feature_penalties must have shape [features, 2]")
    out = np.empty(cf.size, np.float32)
    arrays = [hg, hw, pg, pw, np.ascontiguousarray(cf, np.uint32),
              np.ascontiguousarray(cb, np.uint32), np.ascontiguousarray(types, np.uint8), penalties, out]
    library = ct.CDLL(str(build_probe() if library_path is None else library_path))
    function = library.cbm_scalar_histogram_scores
    function.argtypes = [ct.c_uint32] * 6 + [ct.c_float] + [ct.c_void_p] * 10 + [ct.c_uint32]
    function.restype = ct.c_int
    error = ct.create_string_buffer(4096)
    code = function(rows, features, bins, leaves, cf.size, score_function, l2,
                    *(a.ctypes.data for a in arrays), error, len(error))
    if code:
        raise RuntimeError(error.value.decode())
    return out


@pytest.fixture(scope="module")
def histogram_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        pytest.skip("Requires an Apple Silicon Metal GPU")
    return build_probe()


@pytest.mark.parametrize("left_count", [0, 3, 4, 5, 8])
@pytest.mark.parametrize("tiles", [1, 3])
def test_scalar_histogram_stages_preserve_partition_and_onehot(histogram_library, left_count, tiles):
    # Integer inputs keep every sum exactly representable and isolate layout,
    # numeric-prefix versus one-hot behavior, empty children, and the size tie.
    bins = np.array([[0, 1, 2, 3, 0, 1, 2, 3], [2, 2, 0, 0, 1, 1, 2, 0]], np.uint8)
    g = np.array([7, -4, 3, -2, 1, 5, -6, 8], np.float32)
    w = np.array([1, 0, 2, 4, 3, 1, 2, 1], np.float32)
    left = np.sort(np.array([6, 1, 4, 3, 0, 7, 2, 5])[:left_count])
    right = np.setdiff1d(np.arange(8), left)
    indices = np.r_[left, right]
    offsets = [0, left_count, 8]
    types = [0, 1]
    actual = run_probe(bins, g, w, indices, offsets, types, bin_count=4, tiles=tiles, library_path=histogram_library)
    expected = reference_histograms(bins, g, w, indices, offsets, types, bin_count=4)
    for key in expected:
        if isinstance(expected[key], dict):
            for quantity in ("gradients", "weights"):
                np.testing.assert_array_equal(actual[key][quantity], expected[key][quantity], err_msg=f"{key}/{quantity}")
        else:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def test_invalid_partition_is_rejected_before_loading_metal():
    with pytest.raises(ValueError, match="permutation"):
        run_probe([[0, 1]], [1, -1], [1, 1], [0, 0], [0, 1, 2], [0], library_path="missing")


@pytest.mark.parametrize("active_rows", [(0, 1, 2), (0, 32, 64), (0, 128, 256)])
@pytest.mark.parametrize("values", [(1e8, 1, -1e8), (-1e8, 1, 1e8), (1e8, -1e8, 1)])
@pytest.mark.parametrize("prefix", [False, True])
def test_scalar_histogram_cancellation_preserves_unit_residual(histogram_library, active_rows, values, prefix):
    # Cover one bank, different banks, and repeated row strides through a bank.
    # The stable smaller child gathers all three nonzero rows. The companion
    # feature always collides in one bin; the first optionally spans three bins
    # to exercise prefix expansion separately from local histogram expansion.
    rows = 513
    active = np.array(active_rows)
    bins = np.zeros((2, rows), np.uint8)
    if prefix:
        bins[0, active] = [0, 1, 2]
    g = np.zeros(rows, np.float32)
    g[active] = values
    w = np.ones(rows, np.float32)
    indices = np.r_[active, np.setdiff1d(np.arange(rows), active)]
    offsets = [0, 3, rows]
    args = (bins, g, w, indices, offsets, [0, 0])
    actual = run_probe(*args, bin_count=4, library_path=histogram_library)
    expected = reference_histograms(*args, bin_count=4)
    for stage in _STAGES:
        for quantity in ("gradients", "weights"):
            np.testing.assert_array_equal(actual[stage][quantity], expected[stage][quantity].astype(np.float32),
                                          err_msg=f"{stage}/{quantity}")
    np.testing.assert_array_equal(actual["partition_gradients"], [1, 1, 0])


@pytest.mark.parametrize("score_function", [0, 1, 2, 3])
@pytest.mark.parametrize("candidate_type", [0, 1])
def test_actual_scalar_score_uses_cuda_leaf_precision(histogram_library, score_function, candidate_type):
    # Two complementary numeric orientations exposed a two-ULP Cosine error in
    # the old float calcer. CUDA keeps Cosine accumulators double, but rounds
    # L2's running accumulator after EACH leaf; L2 need not tie this fixture.
    # CUDA feeds the selected histogram first even for one-hot equality bins;
    # its arithmetic order is independent of the model's left/right routing.
    hg = np.array([[-14.683168411254883, 0.6534653306007385],
                   [11.415841102600098, 2.613861322402954]], np.float32)[..., None]
    hw = np.array([[29, 3], [57, 12]], np.float32)[..., None]
    pg = np.array([-14.029703140258789, 14.029702186584473], np.float32)
    pw = np.array([32, 69], np.float32)
    expected = []
    for candidate in range(2):
        l2_score, numerator, denominator = np.float32(0), 0.0, float(np.float32(1e-10))
        for leaf in range(2):
            selected_g, selected_w = hg[leaf, candidate, 0], hw[leaf, candidate, 0]
            children = [(selected_g, selected_w), (np.float32(pg[leaf] - selected_g), pw[leaf] - selected_w)]
            for gradient, weight in children:
                gradient, weight = float(gradient), float(weight)
                mean = gradient / (weight + 3)
                l2_score = np.float32(float(l2_score) - gradient * gradient / (weight + 3))
                numerator += gradient * mean
                denominator += weight * mean * mean
        expected.append(np.float32(-numerator / np.sqrt(denominator)) if score_function & 1 else l2_score)
    actual = score_histograms(hg, hw, pg, pw, [0, 1], [0, 0], [candidate_type] * 2, rows=101,
                              score_function=score_function, library_path=histogram_library)
    np.testing.assert_array_equal(actual, expected)
    if score_function & 1:
        assert actual[0] == actual[1]


@pytest.mark.parametrize("score_function", [0, 1, 2, 3])
def test_redundant_split_orientations_tie_with_deep_empty_leaves(histogram_library, score_function):
    # Adult Logloss's third tree reaches 32 possible leaves but occupies only
    # 12. Sending a whole occupied leaf left or right cannot change its score.
    # Such predicates can still send validation rows to different empty leaves.
    occupied = [0, 9, 16, 18, 22, 24, 25, 26, 27, 28, 29, 31]
    pg, pw = np.zeros(32, np.float32), np.zeros(32, np.float32)
    pg[occupied] = [-0.40910375118255615, 0.25394177436828613,
                    3.0329990684986115, -3.8794389367103577,
                    0.8903173208236694, -1.0470201969146729,
                    6.121322751045227, -0.7142798006534576,
                    2.8230298161506653, -0.40910375118255615,
                    4.55966392159462, 2.0236950367689133]
    pw[occupied] = [1, 1, 12, 11, 2, 3, 26, 2, 11, 1, 19, 12]
    leaves = np.arange(32)
    selected = np.array([leaves < 0, leaves >= 0, leaves % 2 == 0, leaves % 3 == 0]).T
    hg, hw = (selected * pg[:, None])[..., None], (selected * pw[:, None])[..., None]
    score, numerator, denominator = np.float32(0), 0.0, float(np.float32(1e-10))
    for gradient, weight in zip(pg, pw):
        if weight == 0:
            continue
        gradient, weight = float(gradient), float(weight)
        mean = gradient / (weight + 3)
        score = np.float32(float(score) - gradient * mean)
        numerator += gradient * mean
        denominator += weight * mean * mean
    expected = np.float32(-numerator / np.sqrt(denominator)) if score_function & 1 else score
    actual = score_histograms(hg, hw, pg, pw, [0, 1, 2, 3], [0] * 4, [0, 0, 1, 1],
                              rows=101, score_function=score_function, library_path=histogram_library)
    np.testing.assert_array_equal(actual, np.full(4, expected, np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tiles", type=int, default=1)
    parser.add_argument("--no-rebuild", action="store_true")
    args = parser.parse_args()
    if args.compile_only:
        print(build_probe())
        return
    if args.input is None or args.output is None:
        parser.error("provide --input and --output, or --compile-only")
    with np.load(args.input, allow_pickle=False) as data:
        inputs = {name: data[name] for name in ("bins", "gradients", "weights", "row_indices", "partition_offsets", "feature_types")}
        count = int(data["bin_count"]) if "bin_count" in data else None
    result = run_probe(**inputs, bin_count=count, tiles=args.tiles, rebuild=not args.no_rebuild, library_path=args.library)
    flattened = {}
    for name, value in result.items():
        if isinstance(value, dict):
            flattened.update({f"{name}_{quantity}": array for quantity, array in value.items()})
        else:
            flattened[name] = value
    np.savez(args.output, **flattened)
    print(args.output)


if __name__ == "__main__":
    main()
