"""CatBoost model tables for single categorical-feature CUDA-compatible CTRs.

The schema follows ``libs/model/model_export/json_model_helpers.cpp`` and the
inference formulas in ``libs/model/static_ctr_provider.cpp``. The input arrays
are *full training statistics*, not the permutation-prefix statistics used to
train CTR splits. These helpers do not train a model or recompute statistics.
"""

import json
import numbers

import numpy as np


CTR_TYPES = frozenset({"FeatureFreq", "Borders", "Buckets", "FloatTargetMeanValue"})
_HASH_MULTIPLIER = 0x4906BA494954CB65
_UINT64_MASK = (1 << 64) - 1
_MAX_COUNT = np.iinfo(np.int32).max


def _index(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return int(value)


def _hash_array(hashes):
    hashes = np.asarray(hashes)
    if hashes.ndim != 1 or hashes.dtype.kind not in "ui":
        raise ValueError("Category hashes must be a one-dimensional integer array.")
    if np.any(hashes < 0) or np.any(hashes > np.iinfo(np.uint32).max):
        raise ValueError("Category hashes must fit uint32.")
    return hashes.astype(np.uint32, copy=False)


def _float32(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number representable as float32.")
    value = float(value)
    if not np.isfinite(value) or abs(value) > np.finfo(np.float32).max:
        raise ValueError(f"{name} must be a finite number representable as float32.")
    return float(np.float32(value))


def _counts(values, shape, name):
    values = np.asarray(values)
    if values.shape != shape or values.dtype.kind not in "ui":
        raise ValueError(f"{name} must be an integer array with shape {shape}.")
    if np.any(values < 0) or np.any(values > _MAX_COUNT):
        raise ValueError(f"{name} must be nonnegative and fit int32.")
    return values.astype(np.int64)


def combined_category_hashes(hashes):
    """Apply CalcHash(0, (ui64)(int)hash), including signed-32 extension.

    ``libs/model/hash.h`` multiplies modulo 2**64. Sign extension is essential:
    treating a category hash with its high bit set as unsigned changes the CTR
    lookup key and silently sends known categories to the unseen-value prior.
    """
    hashes = _hash_array(hashes)
    values = []
    for value in hashes:
        signed = int(value) if value < (1 << 31) else int(value) - (1 << 32)
        values.append((_HASH_MULTIPLIER * _HASH_MULTIPLIER * signed) & _UINT64_MASK)
    return np.asarray(values, dtype=np.uint64)


def ctr_identifier(cat_feature_index, ctr_type):
    """Return the shared table key used by all priors of a CTR base."""
    if ctr_type not in CTR_TYPES:
        raise ValueError(f"Unsupported CUDA CTR type: {ctr_type}.")
    feature_index = _index(cat_feature_index, "cat_feature_index")
    return json.dumps({"identifier": [{"cat_feature_index": feature_index,
                                      "combination_element": "cat_feature_value"}],
                       "type": ctr_type}, separators=(",", ":"))


def ctr_feature_json(cat_feature_index, ctr_type, borders, *, prior_numerator=0.5,
                     prior_denominator=1.0, target_border_idx=0, shift=0.0, scale=1.0):
    """Make a ``features_info.ctrs`` entry for raw or scaled CTR values.

    The model evaluates ``((numerator + prior_numerator) / (denominator +
    prior_denominator) + shift) * scale`` and tests that result against borders.
    CUDA's CTR scaling must therefore agree with the values quantized by the
    caller. The misspelled ``prior_denomerator`` is the upstream JSON schema.
    """
    identifier = ctr_identifier(cat_feature_index, ctr_type)
    target_border_idx = _index(target_border_idx, "target_border_idx")
    if ctr_type in {"FeatureFreq", "FloatTargetMeanValue"} and target_border_idx != 0:
        raise ValueError(f"{ctr_type} must use target_border_idx=0.")
    prior_numerator = _float32(prior_numerator, "prior_numerator")
    prior_denominator = _float32(prior_denominator, "prior_denominator")
    shift, scale = _float32(shift, "shift"), _float32(scale, "scale")
    if prior_denominator <= 0 or scale <= 0:
        raise ValueError("prior_denominator and scale must be positive.")
    borders = np.asarray(borders)
    if borders.ndim != 1:
        raise ValueError("CTR borders must be one-dimensional.")
    borders = [_float32(value, "CTR border") for value in borders]
    if any(left >= right for left, right in zip(borders, borders[1:])):
        raise ValueError("CTR borders must be strictly increasing after float32 conversion.")
    return {"identifier": identifier, "elements": json.loads(identifier)["identifier"],
            "ctr_type": ctr_type, "prior_numerator": prior_numerator,
            "prior_denomerator": prior_denominator, "shift": shift, "scale": scale,
            "target_border_idx": target_border_idx, "borders": borders}


def ctr_table_json(hashes, counts, *, ctr_type, sums=None, class_counts=None,
                   counter_denominator=None):
    """Make the value stored at ``ctr_data[descriptor['identifier']]``.

    ``hashes`` are distinct original CatBoost uint32 category hashes. ``counts``
    counts unweighted training rows per category, matching CUDA's CTR targets.
    For Borders/Buckets, ``class_counts`` contains all target-bin counts. As a
    binary convenience, ``sums`` may supply positive counts, producing columns
    ``[counts - sums, sums]`` (target index 0 for Borders, 1 for Buckets).

    Current upstream JSON import reads FloatTargetMeanValue sums as integers.
    Fractional sums are rejected explicitly; silently importing them would
    replace them with zero. This restriction belongs to the JSON bridge, not
    the Metal CTR computation. General floating-target means need the native
    model builder or a corrected CatBoost JSON importer.
    """
    if ctr_type not in CTR_TYPES:
        raise ValueError(f"Unsupported CUDA CTR type: {ctr_type}.")
    hashes = _hash_array(hashes)
    if hashes.size == 0 or np.unique(hashes).size != hashes.size:
        raise ValueError("CTR table category hashes must be nonempty and distinct.")
    counts = _counts(counts, hashes.shape, "counts")
    if counter_denominator is not None and ctr_type != "FeatureFreq":
        raise ValueError("counter_denominator applies only to FeatureFreq.")
    denominator = 0
    if ctr_type == "FeatureFreq":
        if sums is not None or class_counts is not None:
            raise ValueError("FeatureFreq accepts counts only.")
        denominator = int(counts.sum()) if counter_denominator is None else _index(
            counter_denominator, "counter_denominator")
        if denominator > _MAX_COUNT or denominator < int(counts.sum()):
            raise ValueError("FeatureFreq denominator must fit int32 and cover all table counts.")
        rows = [[int(value)] for value in counts]
    elif ctr_type == "FloatTargetMeanValue":
        if sums is None or class_counts is not None:
            raise ValueError("FloatTargetMeanValue requires sums and accepts no class_counts.")
        sums = np.asarray(sums)
        if sums.shape != counts.shape or sums.dtype.kind not in "fiu" or not np.all(np.isfinite(sums)):
            raise ValueError("sums must be a finite numeric array matching counts.")
        if np.any(sums != np.trunc(sums)):
            raise ValueError("Fractional FloatTargetMeanValue sums cannot round-trip through CatBoost's JSON importer.")
        if any(not -(1 << 63) <= int(value) < (1 << 63) for value in sums):
            raise ValueError("FloatTargetMeanValue sums must fit int64 for CatBoost JSON import.")
        if np.any((counts == 0) & (sums != 0)):
            raise ValueError("A category with zero count must have zero sum.")
        rows = [[int(value), int(count)] for value, count in zip(sums, counts)]
    else:
        if class_counts is not None:
            if sums is not None:
                raise ValueError("Supply either class_counts or binary sums, not both.")
            shape = np.asarray(class_counts).shape
            if len(shape) != 2 or shape[0] != hashes.size or shape[1] < 2:
                raise ValueError("class_counts must have one row per category and at least two classes.")
            history = _counts(class_counts, shape, "class_counts")
        else:
            if sums is None:
                raise ValueError(f"{ctr_type} requires class_counts or binary sums.")
            sums = np.asarray(sums)
            if (sums.shape != counts.shape or sums.dtype.kind not in "fiu" or
                    not np.all(np.isfinite(sums)) or np.any(sums != np.trunc(sums)) or
                    np.any(sums < 0) or np.any(sums > counts)):
                raise ValueError("Binary sums must be integral positive counts between zero and counts.")
            sums = sums.astype(np.int64)
            history = np.column_stack((counts - sums, sums))
        if not np.array_equal(history.sum(axis=1), counts):
            raise ValueError("class_counts must sum to counts for each category.")
        rows = history.tolist()
    hash_map = []
    for combined_hash, row in zip(combined_category_hashes(hashes), rows):
        hash_map.extend([str(int(combined_hash)), *row])
    return {"hash_map": hash_map, "hash_stride": 1 + len(rows[0]),
            "counter_denominator": denominator}


def online_ctr_split_json(border, split_index, target_border_idx=0):
    """Make a tree split; split_index includes preceding float/one-hot borders."""
    return {"split_type": "OnlineCtr", "border": _float32(border, "CTR border"),
            "ctr_target_border_idx": _index(target_border_idx, "target_border_idx"),
            "split_index": _index(split_index, "split_index")}
