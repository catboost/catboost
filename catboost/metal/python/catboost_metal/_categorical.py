"""CatBoost-compatible host encoding for Metal one-hot categorical splits.

Hashing delegates to CatBoost's own Pool/export implementation, never its
trainer. CatBoost uses CityHash *version 1*, so substituting a current CityHash
package would silently produce incompatible saved models. The native trainer
consumes compact uint8 bins and equality candidates; exported trees retain the
original categorical columns and signed CatBoost hashes.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import numbers
from pathlib import Path
import tempfile

import numpy as np


UNSEEN_CATEGORY_BIN = 255


def _category_strings(values):
    values = np.asarray(values, dtype=object)
    if values.ndim != 1:
        raise ValueError("A categorical feature must be one-dimensional.")
    result = []
    for value in values:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_)):
            result.append(str(int(value)))
        else:
            raise ValueError("Categorical values must be strings or integers; encode missing values as strings.")
    return result


def _hash_unique_categories(values):
    # Repeated small dictionaries benefit from caching. High-cardinality CTR
    # input must not retain entire datasets indefinitely in a process cache.
    return (_cached_category_hashes(values) if len(values) <= 4096
            else _export_category_hashes(values))


@lru_cache(maxsize=128)
def _cached_category_hashes(values):
    return _export_category_hashes(values)


def _export_category_hashes(values):
    """Read CalcCatFeatureHash results through an upstream public export path."""
    from catboost import CatBoostRegressor, Pool

    if not values:
        return ()
    # Saving a model with a Pool exports MergeCatFeaturesHashToString from that
    # Pool even when the constant model has no categorical split. No fit runs.
    skeleton = {
        "features_info": {"categorical_features": [
            {"feature_index": 0, "flat_feature_index": 0, "feature_id": "0"}]},
        "oblivious_trees": [{"splits": [], "leaf_values": [0.0], "leaf_weights": [1.0]}],
        "scale_and_bias": [1.0, [0.0]],
    }
    with tempfile.TemporaryDirectory(prefix="catbooster-category-hash-") as temporary:
        path = Path(temporary) / "hashes.json"
        path.write_text(json.dumps(skeleton))
        model = CatBoostRegressor().load_model(str(path), format="json")
        pool = Pool([[value] for value in values], cat_features=[0])
        model.save_model(str(path), format="json", pool=pool)
        exported = json.loads(path.read_text())["features_info"]["cat_features_hash"]
    mapping = {item["value"]: int(item["hash"]) for item in exported}
    # The upstream map keeps one representative string per hash. Hash each
    # missing spelling on its own to preserve upstream collision semantics.
    # This is rare, but assigning a new bin here would train an invalid model.
    for value in values:
        if value not in mapping:
            if len(values) == 1:
                raise RuntimeError("CatBoost did not export the categorical hash.")
            mapping[value] = _hash_unique_categories((value,))[0]
    return tuple(mapping[value] for value in values)


def cat_feature_hashes(values):
    """Return upstream uint32 hashes for strings and integer categorical values."""
    strings = _category_strings(values)
    unique = tuple(sorted(set(strings)))
    hashes = dict(zip(unique, _hash_unique_categories(unique)))
    return np.asarray([hashes[value] for value in strings], dtype=np.uint32)


@dataclass(frozen=True)
class OneHotEncoding:
    """Perfect-hash dictionary for one original categorical feature."""

    hashes: tuple[int, ...]

    def transform(self, values):
        lookup = {value: index for index, value in enumerate(self.hashes)}
        return np.asarray([lookup.get(int(value), UNSEEN_CATEGORY_BIN)
                           for value in cat_feature_hashes(values)], dtype=np.uint8)

    @property
    def signed_hashes(self):
        return [value if value < (1 << 31) else value - (1 << 32) for value in self.hashes]

    @property
    def candidate_bins(self):
        # A constant categorical feature cannot split the training data.
        return np.arange(len(self.hashes) if len(self.hashes) > 1 else 0, dtype=np.uint32)


def fit_one_hot(values, max_size=255):
    """Return a learned encoding and bins; unseen values always map to 255.

    Values beyond max_size need a CTR path and are rejected explicitly. Integer
    1 and string "1" share a category, exactly as in CatBoost's Pool loader.
    """
    if isinstance(max_size, bool) or not isinstance(max_size, numbers.Integral) or not 1 <= max_size <= 255:
        raise ValueError("one_hot_max_size must be an integer in [1, 255].")
    hashes = cat_feature_hashes(values)
    if hashes.size == 0:
        raise ValueError("Cannot learn categorical values from an empty feature.")
    distinct = tuple(int(value) for value in np.unique(hashes))
    if len(distinct) > max_size:
        raise ValueError(f"Categorical feature has {len(distinct)} values, exceeding one_hot_max_size={max_size}; CTR training is required.")
    encoding = OneHotEncoding(distinct)
    lookup = {value: index for index, value in enumerate(distinct)}
    bins = np.asarray([lookup[int(value)] for value in hashes], dtype=np.uint8)
    return encoding, bins


def categorical_feature_json(feature_index, flat_feature_index, encoding, feature_id=None):
    """Follow model_export/json_model_helpers.cpp's categorical feature schema."""
    result = {"feature_index": int(feature_index), "flat_feature_index": int(flat_feature_index),
              "feature_id": str(flat_feature_index) if feature_id is None else str(feature_id)}
    if len(encoding.hashes) > 1:
        result["values"] = encoding.signed_hashes
    return result


def one_hot_split_json(feature_index, bin_index, encoding, split_index):
    """A true OneHotFeature split: right leaf iff original category matches."""
    if not 0 <= bin_index < len(encoding.hashes):
        raise ValueError("One-hot split refers to an unknown category bin.")
    return {"split_type": "OneHotFeature", "cat_feature_index": int(feature_index),
            "value": encoding.signed_hashes[bin_index], "split_index": int(split_index)}
