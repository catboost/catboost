"""Metal traversal of scalar symmetric CatBoost models, without CPU evaluation.

The CUDA evaluator's double partial accumulators become compensated float pairs
on Metal. GPU traversal and GPU tile reduction return two components; the host
reconstructs double results and applies the model's double scale and bias.
"""

import ctypes as ct
import functools
import hashlib
import numbers
from pathlib import Path
import platform
import subprocess

import numpy as np


class InferenceParams(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in
                ("rows", "features", "trees", "split_stride", "leaf_stride",
                 "tree_start", "tree_end", "batch_rows")] + [
                     ("scale", ct.c_double), ("bias", ct.c_double)]


class InferenceStats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("device_name", ct.c_char * 256)]


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal inference requires macOS on Apple Silicon.")
    import fcntl

    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_inference.h", "metal_inference.mm", "metal_inference_kernels.h"):
        digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_inference_{digest.hexdigest()[:20]}.dylib"
    with (build / "inference_build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                process = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_inference.mm"), "-o", str(temporary),
                ], capture_output=True, text=True, check=False)
                if process.returncode:
                    raise RuntimeError("Could not build Metal inference:\n" + process.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u8, u32, f64 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_double)
    library.cbm_predict_bins.argtypes = [
        ct.POINTER(InferenceParams), u8, ct.c_uint64, u32, ct.c_uint64,
        u32, ct.c_uint64, u32, ct.c_uint64, u8, ct.c_uint64, f64, ct.c_uint64,
        f64, ct.c_uint64, ct.POINTER(InferenceStats), ct.c_char_p, ct.c_size_t,
    ]
    library.cbm_predict_bins.restype = ct.c_int
    return library


def _integer(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def _finite_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite real scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real scalar.")
    return value


def _integer_array(value, name, dimensions, maximum):
    try:
        value = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular integer array.") from error
    if value.ndim != dimensions or value.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a {dimensions}-dimensional integer array.")
    if (value < 0).any() or (value > maximum).any():
        raise ValueError(f"{name} values must be in [0, {maximum}].")
    return value


def predict_bins(bins, depths, split_features, split_bins, leaf_values, *,
                 split_types=None, bias=0.0, scale=1.0, tree_start=0, tree_end=None,
                 batch_size=None, return_stats=False):
    """Predict quantized rows using their actual tree depths on the Metal GPU.

    ``bins`` is feature-major [features, rows]. Split arrays have shape
    [trees, maximum_depth], leaves [trees, leaf_stride], and depths [trees].
    ``split_types`` is 0 (numeric bin > border) or 1 (one-hot bin == category).
    Inactive padded splits are ignored. The tree interval is [start, end), with
    ``None`` selecting all remaining trees. Bias applies only when start is 0.
    Empty rows/ranges return without a GPU dispatch, after validating the model.
    ``return_stats=True`` returns (predictions, statistics); otherwise an ndarray.
    """
    bins = _integer_array(bins, "bins", 2, 255)
    depths = _integer_array(depths, "depths", 1, 16)
    split_features = _integer_array(split_features, "split_features", 2, 2**32 - 1)
    split_bins = _integer_array(split_bins, "split_bins", 2, 255)
    if split_types is None:
        split_types = np.zeros(split_bins.shape, np.uint8)
    split_types = _integer_array(split_types, "split_types", 2, 1)
    features, rows = bins.shape
    trees = depths.size
    if features > 2**32 - 1 or rows > (1 << 27) or trees > 1000000:
        raise ValueError("Inference data or tree count exceeds native limits.")
    if (split_features.shape != split_bins.shape or split_types.shape != split_bins.shape
            or split_features.shape[0] != trees or split_features.shape[1] > 16):
        raise ValueError("Split arrays must have matching [trees, depth<=16] shapes.")
    stride = split_features.shape[1]
    if (depths > stride).any():
        raise ValueError("An actual tree depth exceeds the split stride.")
    try:
        leaf_values = np.asarray(leaf_values)
    except (TypeError, ValueError) as error:
        raise ValueError("leaf_values must be a rectangular real array.") from error
    if (leaf_values.ndim != 2 or leaf_values.dtype.kind not in "biuf"
            or leaf_values.shape[0] != trees or leaf_values.shape[1] < 1):
        raise ValueError("leaf_values must have shape [trees, positive leaf stride].")
    leaf_stride = leaf_values.shape[1]
    if leaf_stride > 2**32 - 1 or (np.left_shift(np.uint64(1), depths.astype(np.uint64)) > leaf_stride).any():
        raise ValueError("Leaf stride cannot contain every actual tree leaf.")
    model_bytes = trees * 4 + split_features.size * 9 + leaf_values.size * 8
    if model_bytes > 1 << 30 or max(split_features.size, leaf_values.size) > 2**32 - 1:
        raise ValueError("Inference model exceeds the 1 GiB memory or GPU index limit.")
    for tree, depth in enumerate(depths):
        depth = int(depth)
        if (split_features[tree, :depth] >= features).any():
            raise ValueError("An active split feature is outside the input feature count.")
        if ((split_types[tree, :depth] == 0) & (split_bins[tree, :depth] >= 255)).any():
            raise ValueError("Numeric split bins must be below 255.")
    scale, bias = _finite_scalar(scale, "scale"), _finite_scalar(bias, "bias")
    tree_start = _integer(tree_start, "tree_start", 0, trees)
    tree_end = trees if tree_end is None else _integer(tree_end, "tree_end", tree_start, trees)
    if tree_end < tree_start:
        raise ValueError("tree_end must be at least tree_start.")
    batch_size = 65536 if batch_size is None else _integer(batch_size, "batch_size", 1, 2**32 - 1)
    if not isinstance(return_stats, (bool, np.bool_)):
        raise ValueError("return_stats must be a boolean.")
    with np.errstate(over="ignore", invalid="ignore"):
        leaf_values = np.ascontiguousarray(leaf_values, dtype=np.float64)
    if (not np.isfinite(leaf_values).all()
            or (np.abs(leaf_values) > np.finfo(np.float32).max / 4).any()):
        raise ValueError("Leaf values must be finite and within the compensated float range.")
    magnitude_bound = sum(float(np.max(np.abs(leaf_values[tree, :1 << int(depths[tree])])))
                          for tree in range(tree_start, tree_end))
    if magnitude_bound > np.finfo(np.float32).max / 4:
        raise ValueError("Leaf magnitudes can overflow the compensated float accumulator.")
    predictions = np.empty(rows, np.float64)
    stats = {"backend": "Metal", "device": "", "kernel_dispatches": 0, "gpu_seconds": 0.0,
             "accumulation": "compensated float pairs"}
    if not rows or tree_start == tree_end:
        predictions.fill(bias if tree_start == 0 else 0.0)
        return (predictions, stats) if return_stats else predictions
    bins = np.ascontiguousarray(bins, dtype=np.uint8)
    depths = np.ascontiguousarray(depths, dtype=np.uint32)
    split_features = np.ascontiguousarray(split_features, dtype=np.uint32)
    split_bins = np.ascontiguousarray(split_bins, dtype=np.uint32)
    split_types = np.ascontiguousarray(split_types, dtype=np.uint8)
    params = InferenceParams(rows, features, trees, stride, leaf_stride,
                             tree_start, tree_end, batch_size, scale, bias)
    native_stats, error = InferenceStats(), ct.create_string_buffer(2048)
    arguments = [ct.byref(params)]
    for array, dtype in ((bins, ct.c_uint8), (depths, ct.c_uint32),
                         (split_features, ct.c_uint32), (split_bins, ct.c_uint32),
                         (split_types, ct.c_uint8), (leaf_values, ct.c_double),
                         (predictions, ct.c_double)):
        arguments.extend([array.ctypes.data_as(ct.POINTER(dtype)), array.size])
    code = _load(build_library()).cbm_predict_bins(
        *arguments, ct.byref(native_stats), error, len(error))
    if code:
        raise RuntimeError("Metal inference failed: " + error.value.decode("utf-8", errors="replace"))
    stats.update(device=native_stats.device_name.decode("utf-8"),
                 kernel_dispatches=int(native_stats.kernel_dispatches),
                 gpu_seconds=float(native_stats.gpu_seconds))
    return (predictions, stats) if return_stats else predictions


def _ctr_values_json(descriptor, table, hashes):
    """Host equivalent of TStaticCtrProvider::CalcCtrs for prepared model tables.

    All arithmetic after integer history accumulation is explicitly float32,
    matching TModelCtr::Calc before strict comparison with float32 borders.
    """
    kind = descriptor.get("ctr_type")
    if kind not in ("FeatureFreq", "Borders", "Buckets", "FloatTargetMeanValue"):
        raise ValueError(f"Unsupported CUDA CTR type in JSON inference: {kind!r}.")
    stride = _integer(table.get("hash_stride"), "CTR hash_stride", 2, 2**31 - 1)
    history = table.get("hash_map")
    if not isinstance(history, list) or len(history) % stride:
        raise ValueError("CTR hash_map length must be a multiple of hash_stride.")
    if ((kind == "FeatureFreq" and stride != 2)
            or (kind == "FloatTargetMeanValue" and stride != 3)
            or (kind in ("Borders", "Buckets") and stride < 3)):
        raise ValueError("CTR hash_stride does not match its history type.")
    target = _integer(descriptor.get("target_border_idx", 0), "CTR target_border_idx", 0,
                      stride - (3 if kind == "Borders" else 2) if kind in ("Borders", "Buckets") else 0)
    denominator = _integer(table.get("counter_denominator", 0), "CTR counter_denominator", 0, 2**31 - 1)
    lookup = {}
    for start in range(0, len(history), stride):
        value = history[start]
        if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
            raise ValueError("CTR table keys must be decimal uint64 strings.")
        key = _integer(int(value), "CTR table key", 0, 2**64 - 1)
        if key in lookup:
            raise ValueError("CTR table keys must be unique.")
        row = history[start + 1:start + stride]
        if kind == "FloatTargetMeanValue":
            numerator = _finite_scalar(row[0], "CTR sum")
            count = _integer(row[1], "CTR count", 0, 2**31 - 1)
            if abs(numerator) > np.finfo(np.float32).max or (count == 0 and numerator != 0):
                raise ValueError("CTR mean sum is invalid or outside the float32 range.")
        else:
            counts = [_integer(item, "CTR class count", 0, 2**31 - 1) for item in row]
            count = sum(counts)
            if count > 2**31 - 1:
                raise ValueError("CTR total count exceeds the int32 history range.")
            if kind == "FeatureFreq":
                numerator, count = counts[0], denominator
                if numerator > denominator:
                    raise ValueError("FeatureFreq count exceeds its denominator.")
            elif kind == "Buckets":
                numerator = counts[target]
            else:
                numerator = sum(counts[target + 1:])
        lookup[key] = (numerator, count)
    missing = (0, denominator if kind == "FeatureFreq" else 0)
    numerator, count = np.zeros(len(hashes), np.float32), np.zeros(len(hashes), np.float32)
    for row, value in enumerate(hashes):
        numerator[row], count[row] = lookup.get(int(value), missing)
    scalars = [_finite_scalar(descriptor.get(name), f"CTR {name}") for name in
               ("prior_numerator", "prior_denomerator", "shift", "scale")]
    if any(abs(value) > np.finfo(np.float32).max for value in scalars):
        raise ValueError("CTR parameters exceed the float32 range.")
    prior_num, prior_denom, shift, scale = map(np.float32, scalars)
    if prior_denom <= 0 or scale <= 0:
        raise ValueError("CTR prior denominator and scale must be positive.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = ((numerator + prior_num) / (count + prior_denom) + shift) * scale
    if not np.isfinite(result).all():
        raise ValueError("CTR calculation overflowed float32.")
    return result


def predict_model_json(model, X, **options):
    """Evaluate parsed scalar symmetric CatBoost JSON on original input values.

    Numeric float32 borders, NaN treatment, flat feature mapping, and categorical
    one-hot equality use the model's existing metadata. Category hashes come
    from CatBoost's upstream hash implementation. More than 255 one-hot values
    occupy multiple byte buckets, each reserving 255 for unmatched values.
    CUDA CTR tables are evaluated during host preparation, then their splits
    run on Metal. Text, embedding and non-symmetric model import remain unsupported.
    ``options`` are the range, batching and statistics options of predict_bins.
    """
    if not isinstance(model, dict) or "oblivious_trees" not in model:
        raise ValueError("A parsed symmetric CatBoost JSON model is required.")
    info = model.get("features_info", {})
    if any(info.get(kind) for kind in ("text_features", "embedding_features")):
        raise ValueError("JSON Metal inference does not support text or embedding models yet.")
    try:
        # Object conversion preserves mixed integer/string categories; ordinary
        # NumPy inference would turn integer categories into floating point.
        data = np.asarray(X, dtype=object)
    except (TypeError, ValueError) as error:
        raise ValueError("X must be a rectangular two-dimensional array.") from error
    if data.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")
    feature_map, categorical_map, category_hashes, flat_features = {}, {}, {}, set()
    bin_columns, grids = [], []
    model_border_count = 0

    def feature_position(feature, mapping):
        index = _integer(feature.get("feature_index"), "feature_index", 0, 2**32 - 1)
        flat = _integer(feature.get("flat_feature_index"), "flat_feature_index", 0, 2**32 - 1)
        if index in mapping or flat in flat_features or flat >= data.shape[1]:
            raise ValueError("Model has duplicate feature positions or X has too few columns.")
        flat_features.add(flat)
        return index, flat

    def check_bin_capacity(additional):
        if (len(bin_columns) + additional) * len(data) > 1 << 30:
            raise ValueError("JSON quantized input exceeds the 1 GiB preparation limit.")

    for feature in info.get("float_features", []):
        index, flat = feature_position(feature, feature_map)
        grid = np.asarray(feature.get("borders", []), np.float32)
        if grid.ndim != 1 or len(grid) > 255 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
            raise ValueError("Model borders must be sorted finite float32 values, at most 255 per feature.")
        treatment = feature.get("nan_value_treatment", "AsIs")
        if treatment not in ("AsIs", "AsFalse", "AsTrue"):
            raise ValueError("Unknown model NaN treatment.")
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                column = np.asarray(data[:, flat], dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Numeric model features must contain real numeric values.") from error
        check_bin_capacity(1)
        quantized = np.searchsorted(grid, column, side="left").astype(np.uint8)
        quantized[np.isnan(column)] = len(grid) if treatment == "AsTrue" else 0
        feature_map[index] = len(bin_columns)
        bin_columns.append(quantized)
        grids.append(grid)
        model_border_count += len(grid)
    for feature in info.get("categorical_features", []):
        from ._categorical import cat_feature_hashes

        index, flat = feature_position(feature, categorical_map)
        values = [_integer(value, "category hash", -(1 << 31), (1 << 32) - 1) & 0xffffffff
                  for value in feature.get("values", [])]
        if len(values) != len(set(values)):
            raise ValueError("Model one-hot category hashes must be unique.")
        bucket_count = (len(values) + 254) // 255
        check_bin_capacity(bucket_count)
        first_bucket = len(bin_columns)
        lookup = {value: (first_bucket + position // 255, position % 255)
                  for position, value in enumerate(values)}
        hashes = cat_feature_hashes(data[:, flat])
        category_hashes[index] = hashes
        buckets = np.full((bucket_count, len(data)), 255, np.uint8)
        for row, value in enumerate(hashes):
            location = lookup.get(int(value))
            if location is not None:
                bucket, category_bin = location
                buckets[bucket - first_bucket, row] = category_bin
        bin_columns.extend(buckets)
        categorical_map[index] = lookup
        model_border_count += len(values)
    ctr_splits = {}
    for descriptor in info.get("ctrs", []):
        import json
        from ._ctr_model import combined_category_hashes

        elements = descriptor.get("elements")
        identifier = descriptor.get("identifier")
        table = model.get("ctr_data", {}).get(identifier)
        if not isinstance(elements, list) or not isinstance(identifier, str) or not isinstance(table, dict):
            raise ValueError("CTR descriptor needs its projection and matching model table.")
        try:
            base = json.loads(identifier)
        except (ValueError, TypeError) as error:
            raise ValueError("CTR identifier must encode its projection and type.") from error
        if base != {"identifier": elements, "type": descriptor.get("ctr_type")}:
            raise ValueError("CTR identifier does not match its descriptor.")
        hashes = np.zeros(len(data), np.uint64)
        multiplier = np.uint64(0x4906BA494954CB65)
        # CatBoost stores the projection as three sequences and combines raw
        # categories first, numeric predicates next, and one-hot predicates last.
        for kind in ("cat_feature_value", "float_feature", "cat_feature_exact_value"):
            for element in elements:
                if element.get("combination_element") not in ("cat_feature_value", "float_feature", "cat_feature_exact_value"):
                    raise ValueError("Unknown CTR projection element.")
                if element.get("combination_element") != kind:
                    continue
                if kind == "cat_feature_value":
                    source = element.get("cat_feature_index")
                    if source not in category_hashes:
                        raise ValueError("CTR projection references an unknown category feature.")
                    contribution = combined_category_hashes(category_hashes[source])
                elif kind == "float_feature":
                    source = element.get("float_feature_index")
                    if source not in feature_map:
                        raise ValueError("CTR projection references an unknown numeric feature.")
                    feature = feature_map[source]
                    border = np.float32(element.get("border"))
                    matches = np.flatnonzero(grids[feature] == border)
                    if len(matches) != 1:
                        raise ValueError("CTR projection border is absent from numeric feature metadata.")
                    contribution = (bin_columns[feature] > matches[0]).astype(np.uint64) * multiplier * multiplier
                else:
                    source = element.get("cat_feature_index")
                    value = _integer(element.get("value"), "CTR one-hot hash", -(1 << 31), 2**32 - 1) & 0xffffffff
                    if source not in category_hashes or value not in categorical_map[source]:
                        raise ValueError("CTR projection references an unknown one-hot value.")
                    contribution = (category_hashes[source] == value).astype(np.uint64) * multiplier * multiplier
                hashes = multiplier * hashes + contribution
        values = _ctr_values_json(descriptor, table, hashes)
        grid = np.asarray(descriptor.get("borders"), np.float32)
        if grid.ndim != 1 or len(grid) > 255 or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
            raise ValueError("CTR borders must be sorted finite float32 values, at most 255 per feature.")
        check_bin_capacity(1)
        slot = len(bin_columns)
        bin_columns.append(np.searchsorted(grid, values, side="left").astype(np.uint8))
        for border_index, border in enumerate(grid):
            ctr_splits[model_border_count + border_index] = (slot, border_index, border,
                                                            descriptor.get("target_border_idx", 0))
        model_border_count += len(grid)
    bins = np.stack(bin_columns) if bin_columns else np.empty((0, len(data)), np.uint8)
    trees = model["oblivious_trees"]
    depths = np.asarray([len(tree.get("splits") or []) for tree in trees], np.uint32)
    stride = int(depths.max(initial=0))
    if stride > 16:
        raise ValueError("Metal inference supports symmetric trees up to depth 16.")
    model_bytes = len(trees) * (4 + stride * 9 + (1 << stride) * 8)
    if len(trees) > 1000000 or model_bytes > 1 << 30:
        raise ValueError("JSON inference model exceeds the 1 GiB memory or tree count limit.")
    split_features = np.zeros((len(trees), stride), np.uint32)
    split_bins = np.zeros_like(split_features)
    split_types = np.zeros((len(trees), stride), np.uint8)
    leaves = np.zeros((len(trees), 1 << stride), np.float64)
    for tree_index, tree in enumerate(trees):
        for level, split in enumerate(tree.get("splits") or []):
            if split.get("split_type") == "FloatFeature":
                source_index = split["float_feature_index"]
                if source_index not in feature_map:
                    raise ValueError("A split references an unknown float feature.")
                feature = feature_map[source_index]
                border = np.float32(split["border"])
                matches = np.flatnonzero(grids[feature] == border)
                if len(matches) != 1:
                    raise ValueError("A split border is absent from its feature quantization grid.")
                split_features[tree_index, level] = feature
                split_bins[tree_index, level] = int(matches[0])
            elif split.get("split_type") == "OneHotFeature":
                source_index = split["cat_feature_index"]
                value = _integer(split["value"], "one-hot split hash", -(1 << 31), (1 << 32) - 1) & 0xffffffff
                location = categorical_map.get(source_index, {}).get(value)
                if location is None:
                    raise ValueError("A one-hot split references a category absent from its feature metadata.")
                split_features[tree_index, level], split_bins[tree_index, level] = location
                split_types[tree_index, level] = 1
            elif split.get("split_type") == "OnlineCtr":
                index = _integer(split.get("split_index"), "CTR split_index", 0, 2**32 - 1)
                if index not in ctr_splits:
                    raise ValueError("An OnlineCtr split_index is absent from the CTR descriptors.")
                feature, border_index, border, target_index = ctr_splits[index]
                if np.float32(split.get("border")) != border or split.get("ctr_target_border_idx") != target_index:
                    raise ValueError("An OnlineCtr split disagrees with its indexed descriptor.")
                split_features[tree_index, level], split_bins[tree_index, level] = feature, border_index
            else:
                raise ValueError("JSON Metal inference supports numeric, one-hot and OnlineCtr splits only.")
        values = np.asarray(tree["leaf_values"], np.float64)
        if values.shape != (1 << int(depths[tree_index]),):
            raise ValueError("Model leaf values must have exactly one scalar per actual leaf.")
        leaves[tree_index, :len(values)] = values
    scale_and_bias = model.get("scale_and_bias", [1.0, [0.0]])
    if len(scale_and_bias) != 2 or len(scale_and_bias[1]) != 1:
        raise ValueError("Metal inference supports scalar scale and bias only.")
    return predict_bins(bins, depths, split_features, split_bins, leaves,
                        split_types=split_types, scale=scale_and_bias[0], bias=scale_and_bias[1][0], **options)
