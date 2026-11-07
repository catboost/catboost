"""Host data preparation using CatBoost's quantizer and categorical hashes."""

from dataclasses import dataclass, field
from functools import lru_cache
import numbers

import numpy as np


class _CudaMersenne64:
    """Exact TRandom engine from util/random/mersenne64.{h,cpp}.

    NumPy's MT19937 engine emits 32-bit values and uses a different shuffle
    protocol. This small host generator preserves CUDA's MT19937-64 sequence;
    category grouping and history scans still execute on Metal.
    """

    _mask = (1 << 64) - 1

    def __init__(self, seed):
        self.state = [int(seed) & self._mask]
        for index in range(1, 312):
            previous = self.state[-1]
            self.state.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & self._mask)
        self.index = 312

    def next(self):
        if self.index == 312:
            for index in range(312):
                value = ((self.state[index] & 0xFFFFFFFF80000000) |
                         (self.state[(index + 1) % 312] & 0x7FFFFFFF))
                self.state[index] = (self.state[(index + 156) % 312] ^ (value >> 1) ^
                                     (0xB5026F5AA96619E9 if value & 1 else 0))
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        return value ^ (value >> 43)

    def advance(self, count):
        for _ in range(count):
            self.next()

    def uniform(self, bound):
        # util/random/common_ops.h::GenUniform excludes RandMax itself even
        # when bound divides 2**64. Keep its exact rejection boundary.
        limit = self._mask - self._mask % bound
        value = self.next()
        while value >= limit:
            value = self.next()
        return value % bound


def _unsigned_integer(value, name, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not 0 <= value <= maximum):
        raise ValueError(f"{name} must be an integer in [0, {maximum}].")
    return int(value)


def cuda_history_order(rows, permutation_id):
    """CUDA DocParallel history order relative to the supplied provider rows.

    The dataset builder uses block size one, independent of the normalized
    fold_permutation_block option. Permutation zero is identity. This helper
    does not perform the separate user-seeded shared Pool preprocessing shuffle
    or group-preserving shuffle; standalone input here is ungrouped row data.
    """
    rows = _unsigned_integer(rows, "rows", 1 << 24)
    permutation_id = _unsigned_integer(permutation_id, "permutation_id", (1 << 32) - 1)
    order = np.arange(rows, dtype=np.uint32)
    if permutation_id:
        seed = (1664525 * permutation_id + 1013904223 + 1) & 0xFFFFFFFF
        random = _CudaMersenne64(seed)
        random.advance(10)
        for index in range(1, rows):
            other = random.uniform(index + 1)
            order[index], order[other] = order[other], order[index]
    return order


@lru_cache(maxsize=128)
def _cuda_iteration_seed(random_seed):
    return _CudaMersenne64(random_seed).next()


def cuda_search_permutation(random_seed, iteration, permutation_count):
    """Choose the structure dataset using CUDA's literal per-iteration rule.

    Its modulo is learnPermutationCount-1, so P=4 selects only 0 or 1 and P=3
    always selects 0. Replacing it with P-1 would change the CUDA algorithm.
    ``iteration`` is absolute, including previously completed snapshot steps.
    """
    random_seed = _unsigned_integer(random_seed, "random_seed", (1 << 64) - 1)
    iteration = _unsigned_integer(iteration, "iteration", (1 << 64) - 1)
    permutation_count = _unsigned_integer(permutation_count, "permutation_count", 64)
    if permutation_count == 0:
        raise ValueError("permutation_count must be an integer in [1, 64].")
    learn_count = permutation_count - 1 if permutation_count > 1 else 1
    if learn_count <= 1:
        return 0
    random = _CudaMersenne64((_cuda_iteration_seed(random_seed) + iteration) & ((1 << 64) - 1))
    random.advance(10)
    return random.next() % (learn_count - 1)


def numeric_array(value, name, dimensions, *, allow_nan=False):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array.") from exc
    if raw.ndim != dimensions or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a {dimensions}-dimensional real numeric array.")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.ascontiguousarray(raw, dtype=np.float32)
    if np.isinf(result).any() or (not allow_nan and np.isnan(result).any()):
        raise ValueError(f"{name} must contain finite float32 values"
                         + (" or NaN." if allow_nan else "; NaN/inf are unsupported."))
    return result


def sample_weights(value, rows, name="sample_weight"):
    if value is None:
        return None
    weights = numeric_array(value, name, 1)
    if weights.shape != (rows,) or (weights < 0).any() or weights.sum(dtype=np.float64) <= 0:
        raise ValueError(f"{name} must contain one nonnegative weight per row and positive total weight.")
    return weights


def quantize_features(features, border_count, nan_mode="Forbidden"):
    """Match CalcQuantizationAndNanMode: reserve one border for training NaNs."""
    from catboost.utils import calculate_quantization_grid

    features = numeric_array(features, "X", 2, allow_nan=nan_mode != "Forbidden")
    if nan_mode not in ("Forbidden", "Min", "Max"):
        raise ValueError("nan_mode must be Forbidden, Min, or Max.")
    borders = []
    bins = np.empty((features.shape[1], features.shape[0]), np.uint8)
    for index in range(features.shape[1]):
        values = features[:, index]
        missing = np.isnan(values)
        budget = border_count - int(missing.any())
        finite = values[~missing]
        grid = np.asarray(calculate_quantization_grid(
            finite, budget, border_type="GreedyLogSum"), dtype=np.float32
        ) if budget > 0 and finite.size else np.empty(0, np.float32)
        if missing.any():
            edge = np.finfo(np.float32).min if nan_mode == "Min" else np.finfo(np.float32).max
            grid = np.concatenate(([np.float32(edge)], grid)) if nan_mode == "Min" else np.append(grid, np.float32(edge))
            grid = np.unique(grid).astype(np.float32)
        if not np.isfinite(grid).all() or (np.diff(grid) <= 0).any() or len(grid) > 255:
            raise RuntimeError("CatBoost returned an invalid numeric quantization grid.")
        borders.append(grid)
        bins[index] = np.searchsorted(grid, values, side="left").astype(np.uint8)
        bins[index, missing] = len(grid) if nan_mode == "Max" else 0
    candidate_features = np.asarray([f for f, grid in enumerate(borders) for _ in grid], np.uint32)
    candidate_bins = np.asarray([b for grid in borders for b in range(len(grid))], np.uint32)
    return borders, bins, candidate_features, candidate_bins


def unpack_pool(X, y=None, weight=None, cat_features=None):
    from catboost import Pool

    names = None
    if isinstance(X, Pool):
        if y is not None or weight is not None:
            raise ValueError("Labels and weights must come from the supplied Pool.")
        if X.get_text_feature_indices() or X.get_embedding_feature_indices():
            raise ValueError("Text and embedding Pool features require the native estimated-feature adapter.")
        pool_cats = X.get_cat_feature_indices()
        if pool_cats:
            raise ValueError("The standalone adapter cannot extract raw categories from Pool; pass a DataFrame/array and cat_features.")
        if X.is_quantized():
            raise ValueError("Prequantized Pool input requires the native CatBoost Metal trainer.")
        y = X.get_label()
        weight = X.get_weight() or None
        names = X.get_feature_names()
        X = X.get_features()
        if cat_features:
            raise ValueError("cat_features does not match the numeric Pool.")
    elif hasattr(X, "columns") and hasattr(X, "to_numpy"):
        names = [str(name) for name in X.columns]
        X = X.to_numpy()
    try:
        array = np.asarray(X)
    except (ValueError, TypeError) as exc:
        raise ValueError("X must be a rectangular feature matrix.") from exc
    if array.ndim != 2 or not all(array.shape):
        raise ValueError("X must contain rows and features.")
    names = names or [str(i) for i in range(array.shape[1])]
    # An unnamed Pool supplies one empty string per column. Preserve explicit
    # names, including names such as "0" that could collide with an index.
    reserved = {name for name in names if name}
    for index, name in enumerate(names):
        if not name:
            fallback = str(index)
            suffix = 1
            while fallback in reserved:
                fallback = f"{index}_{suffix}"
                suffix += 1
            names[index] = fallback
            reserved.add(fallback)
    if len(set(names)) != len(names):
        raise ValueError("Feature names must be unique.")
    indices = []
    for feature in cat_features or []:
        if isinstance(feature, str):
            if feature not in names:
                raise ValueError(f"Unknown categorical feature {feature!r}.")
            feature = names.index(feature)
        if isinstance(feature, bool) or not isinstance(feature, (int, np.integer)) or not 0 <= feature < array.shape[1]:
            raise ValueError("cat_features must contain valid feature indices or names.")
        indices.append(int(feature))
    if len(indices) != len(set(indices)):
        raise ValueError("cat_features contains duplicate features.")
    return array, y, weight, sorted(indices), names


@dataclass
class CtrFeature:
    source_feature: int
    result: object
    target_border: float | None
    random_seed: int


@dataclass
class FeatureLayout:
    borders: list
    has_nans: list
    nan_mode: str
    names: list
    categorical: dict
    ctrs: dict = field(default_factory=dict)
    permutation_bins: tuple = field(default_factory=tuple, repr=False)
    permutation_stats: tuple = field(default_factory=tuple, repr=False)

    @property
    def permutation_count(self):
        return len(self.permutation_bins) or 1

    @property
    def estimation_permutation(self):
        return self.permutation_count - 1

    @property
    def ctr_gpu_stats(self):
        """Aggregate actual CTR preparation calls, including all permutations.

        Each compute_ctr statistic already includes its GPU category sort. The
        top-level totals therefore sum each call once; shared FeatureFreq
        matrices have no repeated calls in later permutation entries.
        """
        calls = [stats for permutation in self.permutation_stats for stats in permutation]
        return {
            "backend": "Metal",
            "device": calls[0].get("device") if calls else None,
            "gpu_seconds": sum(float(stats.get("gpu_seconds", 0)) for stats in calls),
            "kernel_dispatches": sum(int(stats.get("kernel_dispatches", 0)) for stats in calls),
            "compute_calls": len(calls),
            "category_sort_gpu_seconds": sum(float(stats.get("category_sort", {}).get("gpu_seconds", 0))
                                              for stats in calls),
            "category_sort_kernel_dispatches": sum(int(stats.get("category_sort", {}).get("kernel_dispatches", 0))
                                                   for stats in calls),
        }

    def ctr_unique_values(self, heldout=None):
        """Return CUDA model-size cardinality bounds by runtime feature column.

        Numeric and one-hot columns have zero entries. Each simple CTR column
        receives its source category's unique hash count across learn and the
        optional first held-out dataset, capped by their combined row count.
        This follows GetMaxCtrUniqueValues, whose general tensor formula is
        min(2**binary_splits * product(category.OnAll), total_object_count).

        Priors and target configurations remain separate runtime columns, even
        when their cardinality bounds are equal. The caller must maintain used
        flags by these column IDs rather than merging them by category source.
        Held-out categories affect this penalty metadata only; this method does
        not change learned dictionaries, CTR tables, borders, or feature types.
        """
        values = np.zeros(len(self.borders), dtype=np.uint32)
        if not self.ctrs:
            return values
        raw = None
        if heldout is not None:
            if hasattr(heldout, "columns") and hasattr(heldout, "to_numpy"):
                if [str(name) for name in heldout.columns] != self.names:
                    raise ValueError("Feature names/order must match training data.")
                heldout = heldout.to_numpy()
            from catboost import Pool
            if isinstance(heldout, Pool):
                if heldout.get_cat_feature_indices():
                    raise ValueError("The standalone adapter cannot extract raw categories from Pool; pass a DataFrame/array.")
                heldout = heldout.get_features()
            try:
                raw = np.asarray(heldout)
            except (TypeError, ValueError) as exc:
                raise ValueError("Held-out features must be a rectangular matrix.") from exc
            if raw.ndim != 2 or raw.shape[1] != len(self.names):
                raise ValueError(f"Expected {len(self.names)} held-out features.")
        from ._categorical import cat_feature_hashes
        cardinalities = {}
        for feature, ctr in self.ctrs.items():
            source = ctr.source_feature
            if source not in cardinalities:
                hashes = {int(value) for value in ctr.result.hashes}
                learn_rows = int(ctr.result.counts.sum(dtype=np.uint64))
                heldout_rows = 0 if raw is None else len(raw)
                if raw is not None:
                    hashes.update(int(value) for value in cat_feature_hashes(raw[:, source]))
                cardinalities[source] = min(len(hashes), learn_rows + heldout_rows)
            values[feature] = cardinalities[source]
        return values

    def ctr_metadata(self):
        """Include the full learn tables in snapshot/configuration identity."""
        return {str(feature): {
            "source_feature": ctr.source_feature, "type": ctr.result.ctr_type,
            "target_border": ctr.target_border, "random_seed": ctr.random_seed,
            "prior_numerator": ctr.result.prior_numerator,
            "prior_denominator": ctr.result.prior_denominator,
            "hashes": ctr.result.hashes.tolist(), "sums": ctr.result.sums.tolist(),
            "counts": ctr.result.counts.tolist(), "permutation_count": self.permutation_count,
            "permutation_generator": "CatBoostMT19937_64",
            "permutation_block_size": 1, "grid_permutation": 0,
            "estimation_permutation": self.estimation_permutation,
            "history_seeds": [None if permutation == 0 else
                              (1664525 * permutation + 1013904223 + 1) & 0xFFFFFFFF
                              for permutation in range(self.permutation_count)],
        } for feature, ctr in self.ctrs.items()}

    def transform(self, X):
        if hasattr(X, "columns") and hasattr(X, "to_numpy"):
            if [str(name) for name in X.columns] != self.names:
                raise ValueError("Feature names/order must match training data.")
            X = X.to_numpy()
        from catboost import Pool
        if isinstance(X, Pool):
            X = X.get_features()
        raw = np.asarray(X)
        if raw.ndim != 2 or raw.shape[1] != len(self.names):
            raise ValueError(f"Expected {len(self.names)} features.")
        bins = np.empty((len(self.borders), raw.shape[0]), np.uint8)
        ctr_sources = {ctr.source_feature for ctr in self.ctrs.values()}
        category_hashes = {}
        for feature, grid in enumerate(self.borders):
            if feature in self.ctrs:
                from ._categorical import cat_feature_hashes
                ctr = self.ctrs[feature]
                if ctr.source_feature not in category_hashes:
                    category_hashes[ctr.source_feature] = cat_feature_hashes(raw[:, ctr.source_feature])
                values = ctr.result.full_values(category_hashes[ctr.source_feature])
                bins[feature] = np.searchsorted(grid, values, side="left").astype(np.uint8)
                continue
            if feature in ctr_sources:
                # Keep the original column's runtime slot so raw feature ids
                # remain stable. Only its generated CTR columns have splits.
                bins[feature] = 0
                continue
            if feature in self.categorical:
                bins[feature] = self.categorical[feature].transform(raw[:, feature])
                continue
            column = raw[:, feature]
            if self.categorical:
                try:
                    column = column.astype(np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {feature} must be numeric.") from exc
            values = numeric_array(column, "X", 1, allow_nan=self.nan_mode != "Forbidden")
            bins[feature] = np.searchsorted(grid, values, side="left").astype(np.uint8)
            # Without training NaNs, CatBoost's stored treatment is AsIs (left).
            bins[feature, np.isnan(values)] = len(grid) if self.has_nans[feature] and self.nan_mode == "Max" else 0
        return bins


def prepare_features(raw, border_count, nan_mode, categorical_indices, names, one_hot_max_size,
                     *, targets=None, objective="RMSE", ctr_type="Borders", ctr_prior=0.5,
                     ctr_target_border=None, random_seed=0, ctr_border_count=15, permutation_count=1,
                     one_hot_cardinalities=None, history_orders=None, ctr_group_ids=None):
    """Prepare shared candidates and same-shape per-permutation CTR datasets.

    The returned bins remain permutation zero for existing callers. Every
    dataset is available as ``layout.permutation_bins``; target and weight row
    order is identical for all of them. CTR borders and full inference tables
    come from permutation zero and are reused across datasets. Numeric/one-hot
    inputs collapse to one dataset. Scalar and multiclass objectives use the
    same feature grids while their runtimes maintain separate cursor dimensions.
    """
    if ctr_type not in ("Borders", "FeatureFreq"):
        raise ValueError("Standalone CTR training currently supports Borders or FeatureFreq.")
    if (isinstance(random_seed, (bool, np.bool_)) or not isinstance(random_seed, numbers.Integral)
            or not 0 <= random_seed < (1 << 64)):
        raise ValueError("random_seed must be an integer in [0, 2^64).")
    if (isinstance(ctr_border_count, bool) or not isinstance(ctr_border_count, numbers.Integral)
            or not 1 <= ctr_border_count <= 255):
        raise ValueError("ctr_border_count must be an integer in [1, 255].")
    if (isinstance(ctr_prior, (bool, np.bool_)) or not isinstance(ctr_prior, numbers.Real)
            or not np.isfinite(ctr_prior) or abs(ctr_prior) > np.finfo(np.float32).max):
        raise ValueError("ctr_prior must be a finite float32 number.")
    if ctr_target_border is not None and (
            isinstance(ctr_target_border, (bool, np.bool_)) or not isinstance(ctr_target_border, numbers.Real)
            or not np.isfinite(ctr_target_border) or abs(ctr_target_border) > np.finfo(np.float32).max):
        raise ValueError("ctr_target_border must be a finite float32 number or None.")
    if ctr_type == "FeatureFreq" and ctr_target_border is not None:
        raise ValueError("FeatureFreq does not use a target border.")
    permutation_count = _unsigned_integer(permutation_count, "permutation_count", 64)
    if permutation_count == 0:
        raise ValueError("permutation_count must be an integer in [1, 64].")
    if history_orders is not None:
        orders = np.asarray(history_orders)
        if (orders.shape != (permutation_count, raw.shape[0]) or orders.dtype.kind not in "iu"
                or (orders < 0).any() or (orders >= raw.shape[0]).any()
                or any(not np.all(np.bincount(order, minlength=raw.shape[0]) == 1) for order in orders)):
            raise ValueError("CTR history orders must contain each original row exactly once per permutation.")
        history_orders = np.ascontiguousarray(orders, np.uint32)
    if one_hot_cardinalities is not None:
        if not isinstance(one_hot_cardinalities, dict) or set(one_hot_cardinalities) != set(categorical_indices):
            raise ValueError("One-hot cardinalities must specify each categorical feature.")
        one_hot_cardinalities = {feature: _unsigned_integer(count, "one-hot cardinality", 2**32-1)
                                for feature, count in one_hot_cardinalities.items()}
    borders, has_nans, categorical = [], [], {}
    bins = np.empty((raw.shape[1], raw.shape[0]), np.uint8)
    candidates, thresholds, types = [], [], []
    pending_ctrs = []
    for feature in range(raw.shape[1]):
        if feature in categorical_indices:
            from ._categorical import fit_one_hot, cat_feature_hashes, OneHotEncoding
            hashes = cat_feature_hashes(raw[:, feature])
            count = np.unique(hashes).size
            on_all = count if one_hot_cardinalities is None else one_hot_cardinalities[feature]
            if on_all < count:
                raise ValueError("One-hot cardinality cannot be smaller than the learn dictionary.")
            if on_all > one_hot_max_size:
                # High-cardinality data is represented by actual OnlineCtr
                # splits; it is never silently expanded to numeric one-hot.
                encoding, values = OneHotEncoding(()), np.zeros(raw.shape[0], np.uint8)
                pending_ctrs.append((feature, hashes))
            else:
                encoding, values = fit_one_hot(raw[:, feature], max_size=one_hot_max_size)
            categorical[feature] = encoding
            borders.append(np.empty(0, np.float32))
            has_nans.append(False)
            bins[feature] = values
            count = len(encoding.candidate_bins)
            candidates.extend([feature] * count)
            thresholds.extend(encoding.candidate_bins.tolist())
            types.extend([1] * count)
        else:
            column = raw[:, feature]
            if categorical_indices:
                try:
                    column = column.astype(np.float32)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Feature {feature} must be numeric.") from exc
            values = numeric_array(column, "X", 1, allow_nan=nan_mode != "Forbidden")
            grid, codes, _, thresholds_for_feature = quantize_features(values[:, None], border_count, nan_mode)
            borders.append(grid[0])
            has_nans.append(bool(np.isnan(values).any()))
            bins[feature] = codes[0]
            candidates.extend([feature] * len(grid[0]))
            thresholds.extend(thresholds_for_feature.tolist())
            types.extend([0] * len(grid[0]))
    ctrs = {}
    permutation_bins = None
    permutation_stats = []
    if pending_ctrs:
        if (raw.shape[1] + len(pending_ctrs)) * raw.shape[0] * permutation_count > (1 << 30):
            raise ValueError("Categorical permutation feature matrices exceed the 1 GiB limit.")
        from catboost.utils import calculate_quantization_grid
        from ._ctrs import compute_ctr
        target_border = None
        ctr_targets = None
        if ctr_type == "Borders":
            if targets is None:
                raise ValueError("Borders CTR training requires targets.")
            targets = numeric_array(targets, "CTR targets", 1)
            if targets.shape != (raw.shape[0],):
                raise ValueError("CTR targets must contain one value per training row.")
            if ctr_target_border is not None:
                target_border = float(np.float32(ctr_target_border))
            elif objective in ("Logloss", "CrossEntropy"):
                target_border = 0.5
            else:
                grid = calculate_quantization_grid(targets, 1, border_type="MinEntropy")
                target_border = float(np.float32(grid[0] if grid else targets[0]))
            ctr_targets = (targets > target_border).astype(np.float32)
        permutation = cuda_history_order(raw.shape[0], 0) if history_orders is None else history_orders[0]
        generated_bins = []
        first_stats = []
        for source_feature, hashes in pending_ctrs:
            result = compute_ctr(hashes, ctr_targets, ctr_type=ctr_type, permutation=permutation,
                                 prior_numerator=ctr_prior, prior_denominator=1.0, group_ids=ctr_group_ids)
            grid = np.asarray(calculate_quantization_grid(
                result.values, int(ctr_border_count), border_type="Uniform"), dtype=np.float32)
            if not grid.size:
                # TGpuBordersBuilder preserves a candidate for constant CTRs.
                grid = np.asarray([0.5], dtype=np.float32)
            if not np.isfinite(grid).all() or (np.diff(grid) <= 0).any() or len(grid) > 255:
                raise RuntimeError("CatBoost returned an invalid CTR quantization grid.")
            feature = len(borders)
            borders.append(grid)
            has_nans.append(False)
            generated_bins.append(np.searchsorted(grid, result.values, side="left").astype(np.uint8))
            candidates.extend([feature] * len(grid))
            thresholds.extend(range(len(grid)))
            types.extend([0] * len(grid))
            ctrs[feature] = CtrFeature(source_feature, result, target_border, int(random_seed))
            first_stats.append(dict(result.stats))
        bins = np.concatenate((bins, np.asarray(generated_bins, dtype=np.uint8)), axis=0)
        permutation_bins = [bins]
        permutation_stats.append(tuple(first_stats))
        for permutation_id in range(1, permutation_count):
            if ctr_type == "FeatureFreq":
                # Frequency CTRs are permutation independent; share the same
                # immutable-by-convention matrix instead of recomputing them.
                permutation_bins.append(bins)
                permutation_stats.append(())
                continue
            order = cuda_history_order(raw.shape[0], permutation_id) if history_orders is None else history_orders[permutation_id]
            current_bins = bins.copy()
            current_stats = []
            for (feature, ctr), (_, hashes) in zip(ctrs.items(), pending_ctrs):
                result = compute_ctr(hashes, ctr_targets, ctr_type=ctr_type, permutation=order,
                                     prior_numerator=ctr_prior, prior_denominator=1.0, group_ids=ctr_group_ids)
                current_bins[feature] = np.searchsorted(borders[feature], result.values, side="left").astype(np.uint8)
                current_stats.append(dict(result.stats))
            permutation_bins.append(current_bins)
            permutation_stats.append(tuple(current_stats))
    layout = FeatureLayout(borders, has_nans, nan_mode, names, categorical, ctrs)
    layout.permutation_bins = tuple(permutation_bins) if permutation_bins is not None else (bins,)
    layout.permutation_stats = tuple(permutation_stats) if permutation_stats else ((),)
    return layout, bins, np.asarray(candidates, np.uint32), np.asarray(thresholds, np.uint32), np.asarray(types, np.uint8)
