"""Persistent numeric/one-hot Ordered boosting on Metal with independent prefix cursors."""
import ctypes as ct
import functools
import hashlib
import json
import numbers
from pathlib import Path
import platform
import re
import subprocess
import threading

import numpy as np

from ._ordered_rng import OrderedSelectionRng, cuda_ordered_block_size, cuda_ordered_history_order, cuda_ordered_group_history_order
from ._native import (TrainResult, StepResult, StepInfo, StructureInfo, BootstrapOptions, ScoreNoiseOptions,
                     FeaturePenaltyOptions, QueryOptions, PairOptions, YetiRankOptions, _u8, _u32, _f32, _stats)


class Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "candidates", "iterations", "depth", "objective", "score_function", "leaf_method",
        "leaf_iterations", "permutations", "min_fold_size", "normalize")]
    _fields_ += [(name, ct.c_float) for name in ("learning_rate", "l2", "bias", "fold_growth", "objective_param")]
    _fields_ += [(name, ct.c_uint32) for name in ("reserved0", "reserved1", "reserved2")]


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Ordered Metal training requires macOS on Apple Silicon.")
    import fcntl
    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    names = ("metal_ordered_trainer.mm", "metal_ordered_trainer.h", "metal_ordered_kernels.h",
             "metal_ordered_session_kernels.h", "metal_trainer.h", "metal_bootstrap_kernels.h",
             "metal_score_noise_kernels.h", "metal_kernels.h", "metal_kernel_abi.h", "metal_additional_objective_kernels.h",
             "metal_objective_kernels.h", "metal_backtracking_kernels.h", "metal_ordered_backtracking.h",
             "metal_deep_partition_kernels.h", "metal_ordered_histogram_kernels.h", "metal_ordered_histogram_runtime.h",
             "metal_exact_leaf_kernels.h", "metal_sort.mm", "metal_sort.h", "metal_sort_kernels.h",
             "metal_querywise_kernels.h", "metal_pairwise_kernels.h", "metal_ordered_query_kernels.h",
             "metal_ordered_yeti_runtime.h", "metal_yeti_rank_kernels.h")
    # Keep the established hash order, then include every transitive local
    # header so changes to newly integrated targets invalidate the cache too.
    explicit = [native / name for name in names]
    pending, sources = explicit.copy(), {}
    while pending:
        source = pending.pop()
        if source in sources:
            continue
        data = source.read_bytes()
        sources[source] = data
        for name in re.findall(rb'^\s*#include\s+"([^"\n]+)"', data, flags=re.MULTILINE):
            dependency = source.parent / name.decode()
            if dependency.is_file():
                pending.append(dependency)
    digest = hashlib.sha256(platform.platform().encode())
    for source in explicit + sorted(sources.keys() - set(explicit)):
        digest.update(source.name.encode())
        digest.update(sources[source])
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_ordered_{digest.hexdigest()[:20]}.dylib"
    with (build / "ordered-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                result = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal", str(native / "metal_ordered_trainer.mm"), str(native / "metal_sort.mm"),
                    "-o", str(temporary)], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError("Could not build Ordered Metal runtime:\n" + result.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    error = [ct.c_char_p, ct.c_size_t]
    library.cbm_ordered_session_create.argtypes = [ct.POINTER(Params), u8, f32, f32, f32, u32, u32, u32,
                                                  ct.POINTER(ct.c_void_p)] + error
    library.cbm_ordered_session_create_typed.argtypes = [ct.POINTER(Params), u8, f32, f32, f32, u32, u32, u8, u32,
                                                        ct.POINTER(ct.c_void_p)] + error
    library.cbm_ordered_session_create_typed.restype = ct.c_int
    library.cbm_ordered_session_create_grouped.argtypes = [ct.POINTER(Params), u8, f32, f32, f32, u32, u32, u8, u32,
                                                          ct.c_uint32, u32, ct.c_double, ct.POINTER(ct.c_void_p)] + error
    library.cbm_ordered_session_create_grouped.restype = ct.c_int
    library.cbm_ordered_session_create_banked.argtypes = [ct.POINTER(Params), ct.c_uint32, ct.c_uint64,
        u8, f32, f32, f32, u32, u32, u8, u32, ct.c_uint32, u32, ct.c_double, ct.POINTER(ct.c_void_p)] + error
    library.cbm_ordered_session_create_banked.restype = ct.c_int
    ranked_prefix = [ct.POINTER(Params), ct.c_uint32, ct.c_uint64, u8]
    ranked_data = [f32, f32, f32, u32, u32, u8, u32]
    ranked_tail = [u32, ct.c_double, ct.POINTER(ct.c_void_p)] + error
    library.cbm_ordered_session_create_query_banked.argtypes = ranked_prefix + ranked_data + [ct.POINTER(QueryOptions)] + ranked_tail
    library.cbm_ordered_session_create_query_banked.restype = ct.c_int
    library.cbm_ordered_session_create_pair_banked.argtypes = ranked_prefix + [f32, u32, u32, u8, u32,
        ct.POINTER(PairOptions), u32, u32, f32] + ranked_tail
    library.cbm_ordered_session_create_pair_banked.restype = ct.c_int
    library.cbm_ordered_session_create_yeti_banked.argtypes = ranked_prefix + ranked_data + [ct.POINTER(YetiRankOptions)] + ranked_tail
    library.cbm_ordered_session_create_yeti_banked.restype = ct.c_int
    library.cbm_ordered_session_set_feature_penalties.argtypes = [ct.c_void_p, ct.POINTER(FeaturePenaltyOptions), u32, f32] + error
    library.cbm_ordered_session_set_feature_penalties.restype = ct.c_int
    library.cbm_ordered_session_step.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(StepInfo), u32,
                                                u32, u32, u8, f32, f32] + error
    library.cbm_ordered_session_begin_tree.argtypes = [ct.c_void_p, ct.c_uint32] + error
    library.cbm_ordered_session_begin_tree.restype = ct.c_int
    library.cbm_ordered_session_grow_tree.argtypes = [ct.c_void_p, ct.POINTER(StructureInfo)] + error
    library.cbm_ordered_session_grow_tree.restype = ct.c_int
    library.cbm_ordered_session_finish_tree.argtypes = [ct.c_void_p, ct.POINTER(StepInfo), u32, u32, u32, u8, f32, f32] + error
    library.cbm_ordered_session_finish_tree.restype = ct.c_int
    for name in ("set_yeti_oracle_seeds", "set_yeti_leaf_seeds"):
        operation = getattr(library, "cbm_ordered_session_" + name)
        operation.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(ct.c_uint64)] + error
        operation.restype = ct.c_int
    library.cbm_ordered_session_yeti_seed_shape.argtypes = [ct.c_void_p, ct.c_uint32, u32, u32] + error
    library.cbm_ordered_session_yeti_seed_shape.restype = ct.c_int
    library.cbm_ordered_session_info.argtypes = [ct.c_void_p, ct.POINTER(StepInfo)] + error
    library.cbm_ordered_session_copy_predictions.argtypes = [ct.c_void_p, f32] + error
    library.cbm_ordered_session_set_bootstrap.argtypes = [ct.c_void_p, ct.POINTER(BootstrapOptions), ct.c_uint32] + error
    library.cbm_ordered_session_set_score_noise.argtypes = [ct.c_void_p, ct.POINTER(ScoreNoiseOptions)] + error
    library.cbm_ordered_session_set_backtracking.argtypes = [ct.c_void_p, ct.c_uint32] + error
    library.cbm_ordered_session_get_bootstrap_state.argtypes = [ct.c_void_p, u32, f32, u32] + error
    library.cbm_ordered_session_state_shape.argtypes = [ct.c_void_p, u32, u32] + error
    library.cbm_ordered_session_copy_state.argtypes = [ct.c_void_p, ct.c_uint32, ct.c_uint32, u32, f32] + error
    library.cbm_ordered_session_restore_cursors.argtypes = [ct.c_void_p, ct.c_uint32, f32] + error
    library.cbm_ordered_session_close.argtypes = [ct.c_void_p]
    library.cbm_ordered_session_close.restype = None
    for name in ("create", "step", "info", "copy_predictions", "state_shape", "copy_state", "restore_cursors", "set_bootstrap", "set_score_noise", "set_backtracking", "get_bootstrap_state"):
        getattr(library, "cbm_ordered_session_" + name).restype = ct.c_int
    return library


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low},{high}].")
    return int(value)


def _finite(name, value, shape):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or array.shape != shape:
        raise ValueError(f"{name} must be a numeric array with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite float32 values.")
    return array


def _check(code, error):
    if code:
        raise RuntimeError("Metal Ordered training failed: " + error.value.decode("utf-8", errors="replace"))


class Session:
    """Numeric/one-hot Ordered training with independent fold and full-model cursors.

    Inputs use the scalar feature-major quantized ABI. ``state()`` captures
    every prefix cursor, so ``initial_state`` can resume the actual trajectory.
    ``initial_predictions`` alone initializes a new sequence from a model.
    """
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, iterations=100,
                 depth=6, learning_rate=.03, l2_leaf_reg=3., bias=0., objective="RMSE",
                 score_function="Cosine", sample_weight=None, leaf_estimation_method="Newton",
                 leaf_estimation_iterations=1, leaf_estimation_backtracking="No", initial_predictions=None,
                 candidate_types=None, permutation_count=1, permutations=None, fold_len_multiplier=2.,
                 min_fold_size=100, fold_size_loss_normalization=False, random_seed=0, iteration_offset=0,
                 initial_state=None, bootstrap_type="No", random_strength=0., objective_param=None,
                 bagging_temperature=1., subsample=1., mvs_reg=None, initial_mvs_lambda=None,
                 observations_to_bootstrap="TestOnly", fold_permutation_block=64, group_sizes=None, permutation_bins=None,
                 ctr_unique_values=None, model_size_reg=.5, feature_weights=None, group_offsets=None,
                 query_beta=1., query_lambda=.01, pair_winners=None, pair_losers=None, pair_weights=None,
                 yeti_permutations=10, decay=.85, legacy_prefix_centering=False, subgroup_hashes=None,
                 simple_ctr_permutation_dependent=False):
        self._handle, self._lib, self._lock = ct.c_void_p(), None, threading.RLock()
        self._steps = []
        iterations = _integer("iterations", iterations, 1, 100000)
        depth = _integer("depth", depth, 0, 16)
        leaf_steps = _integer("leaf_estimation_iterations", leaf_estimation_iterations, 1, 1000)
        permutation_count = _integer("permutation_count", permutation_count, 1, 64)
        minimum = _integer("min_fold_size", min_fold_size, 1, 2**32 - 1)
        self._random_seed = _integer("random_seed", random_seed, 0, 2**64 - 1)
        self._iteration_offset = _integer("iteration_offset", iteration_offset, 0, 2**32 - 1 - iterations)
        objectives = dict(zip(("RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber", "Expectile", "Lq", "Tweedie",
                               "LogLinQuantile", "Quantile", "MAE", "MAPE", "QueryRMSE", "QuerySoftMax", "PairLogit"), range(15)))
        objectives["YetiRank"] = 17
        if objective not in objectives:
            raise ValueError("Unsupported Ordered objective.")
        queried = objective in ("QueryRMSE", "QuerySoftMax")
        paired, yeti = objective == "PairLogit", objective == "YetiRank"
        ranking = queried or paired or yeti
        self._yeti = yeti
        if not isinstance(simple_ctr_permutation_dependent, (bool, np.bool_)):
            raise ValueError("simple_ctr_permutation_dependent must be boolean.")
        if simple_ctr_permutation_dependent and not yeti:
            raise ValueError("Explicit CTR seed scheduling is supported for Ordered YetiRank only.")
        if not paired and any(value is not None for value in (pair_winners, pair_losers, pair_weights)):
            raise ValueError("Supplied pair arrays require Ordered PairLogit.")
        if paired and sample_weight is not None:
            raise ValueError("Ordered PairLogit uses incident pair mass; sample_weight must be None.")
        if not queried and (query_beta != 1. or query_lambda != .01):
            raise ValueError("Query parameters require Ordered QueryRMSE or QuerySoftMax.")
        query_parameters = _finite("query parameters", [query_beta, query_lambda], (2,))
        if yeti:
            yeti_permutations = _integer("yeti_permutations", yeti_permutations, 1, 10000)
            decay = float(_finite("decay", decay, ()))
            if not 0 <= decay <= 1:
                raise ValueError("YetiRank decay must be in [0,1].")
            if not isinstance(legacy_prefix_centering, (bool, np.bool_)):
                raise ValueError("YetiRank legacy_prefix_centering must be boolean.")
            if leaf_estimation_method != "Newton" or leaf_estimation_backtracking != "No":
                raise ValueError("Ordered YetiRank requires Newton leaves and no backtracking like CUDA.")
        elif yeti_permutations != 10 or decay != .85 or legacy_prefix_centering or subgroup_hashes is not None:
            raise ValueError("YetiRank options require the Ordered YetiRank objective.")
        if score_function not in ("Cosine", "NewtonCosine"):
            raise ValueError("Ordered supports Cosine and NewtonCosine scoring.")
        if leaf_estimation_method not in ("Newton", "Gradient", "Exact"):
            raise ValueError("Ordered supports Newton/Gradient/Exact leaf estimation.")
        if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
            raise ValueError("Unknown Ordered leaf_estimation_backtracking rule.")
        if leaf_estimation_method == "Exact" and objective not in ("Quantile", "MAE", "MAPE"):
            raise ValueError("Ordered Exact supports Quantile, MAE and MAPE only.")
        if bootstrap_type not in ("No", "Bayesian", "Bernoulli", "Poisson", "MVS"):
            raise ValueError("Unknown Ordered bootstrap type.")
        sampling = _finite("sampling parameters", [bagging_temperature, subsample, random_strength], (3,))
        if sampling[0] < 0 or not 0 < sampling[1] <= 1 or sampling[2] < 0 or (bootstrap_type == "Poisson" and sampling[1] == 1):
            raise ValueError("Invalid Ordered bootstrap parameters or random_strength.")
        regularization = None if mvs_reg is None else float(_finite("mvs_reg", mvs_reg, ()))
        initial_lambda = None if initial_mvs_lambda is None else float(_finite("initial_mvs_lambda", initial_mvs_lambda, ()))
        if (regularization is not None and regularization < 0) or (initial_lambda is not None and initial_lambda < 0):
            raise ValueError("Ordered MVS regularization must be nonnegative.")
        if observations_to_bootstrap not in ("TestOnly", "LearnAndTest"):
            raise ValueError("observations_to_bootstrap must be TestOnly or LearnAndTest.")
        if objective in ("Huber", "Expectile", "Lq", "Tweedie") and objective_param is None:
            raise ValueError(f"{objective} requires an explicit objective_param.")
        parameter = float(_finite("objective_param", (.5 if objective in ("LogLinQuantile", "Quantile", "MAE") else 1)
                                 if objective_param is None else objective_param, ()))
        if ((objective == "Huber" and parameter < 0) or
                (objective in ("Expectile", "LogLinQuantile", "Quantile") and not 0 <= parameter <= 1) or
                (objective == "Lq" and parameter < 1) or (objective == "Tweedie" and not 1 < parameter < 2)):
            raise ValueError("Invalid Ordered objective_param.")
        if leaf_estimation_method == "Newton" and (objective in ("LogLinQuantile", "Quantile", "MAE", "MAPE") or
                                                     (objective == "Lq" and parameter < 2)):
            raise ValueError("Newton leaf estimation is unsupported for this Ordered objective.")
        if not isinstance(fold_size_loss_normalization, (bool, np.bool_)):
            raise ValueError("fold_size_loss_normalization must be boolean.")
        bins = np.asarray(bins)
        if bins.ndim != 2 or bins.dtype.kind not in "iu" or not all(bins.shape):
            raise ValueError("bins must be a nonempty feature-major integer matrix.")
        features, rows = bins.shape
        _integer("rows", rows, 4, 1 << 24)
        block_size = cuda_ordered_block_size(rows, fold_permutation_block)
        if rows * features > 2**32 - 1 or (bins < 0).any() or (bins > 255).any():
            raise ValueError("bins must be in [0,255] with at most uint32 cells.")
        bins = np.ascontiguousarray(bins, np.uint8)
        if permutation_bins is not None:
            banks = np.asarray(permutation_bins)
            if (banks.shape != (permutation_count, features, rows) or banks.dtype.kind not in "iu"
                    or (banks < 0).any() or (banks > 255).any() or banks.size > (1 << 30)):
                raise ValueError("permutation_bins must be uint8-range [permutation_count,features,rows] within 1 GiB.")
            if not np.array_equal(banks[0], bins):
                raise ValueError("permutation_bins bank zero must equal bins.")
            # Shared banks retain the preceding numeric/one-hot snapshot identity.
            permutation_bins = None if all(np.array_equal(bank, bins) for bank in banks) else np.ascontiguousarray(banks, np.uint8)
        targets = np.zeros(rows, np.float32) if paired and targets is None else _finite("targets", targets, (rows,))
        if objective == "Logloss" and ((targets != 0) & (targets != 1)).any():
            raise ValueError("Logloss targets must be zero or one.")
        if objective == "CrossEntropy" and ((targets < 0) | (targets > 1)).any():
            raise ValueError("CrossEntropy targets must be in [0,1].")
        if objective in ("Poisson", "Tweedie") and (targets < 0).any():
            raise ValueError("Poisson/Tweedie targets must be nonnegative.")
        if yeti and ((targets < 0).any() or (targets > 1).any()):
            raise ValueError("Classic YetiRank with PFound requires relevance labels in [0,1].")
        weights = np.ones(rows, np.float32) if sample_weight is None else _finite("sample_weight", sample_weight, (rows,))
        if (weights < 0).any() or not 0 < weights.sum(dtype=np.float64) < 1e30:
            raise ValueError("sample_weight must be nonnegative with positive total below 1e30.")
        cf, cb = np.asarray(candidate_features), np.asarray(candidate_bins)
        if cf.ndim != 1 or cf.shape != cb.shape:
            raise ValueError("Candidate arrays must be matching vectors.")
        if cf.size and (cf.dtype.kind not in "iu" or cb.dtype.kind not in "iu" or
                       (cf < 0).any() or (cf >= features).any() or (cb < 0).any() or (cb > 255).any()):
            raise ValueError("Invalid Ordered numeric/one-hot candidate.")
        candidate_types = np.zeros(cf.size, np.uint8) if candidate_types is None else np.asarray(candidate_types)
        if (candidate_types.shape != cf.shape or candidate_types.dtype.kind not in "biu"
                or ((candidate_types != 0) & (candidate_types != 1)).any()):
            raise ValueError("Ordered candidate_types must contain numeric(0)/one-hot(1) flags.")
        if ((candidate_types == 0) & (cb == 255)).any():
            raise ValueError("Invalid Ordered numeric candidate border 255.")
        feature_kinds = {}
        for feature, kind in zip(cf, candidate_types):
            if feature_kinds.setdefault(int(feature), int(kind)) != int(kind):
                raise ValueError("An Ordered feature cannot mix numeric and one-hot candidates.")
        cf, cb = np.ascontiguousarray(cf, np.uint32), np.ascontiguousarray(cb, np.uint32)
        candidate_types = np.ascontiguousarray(candidate_types, np.uint8)
        if group_offsets is not None:
            from ._query_data import validate_offsets
            group_offsets = validate_offsets(group_offsets, rows)
            offset_sizes = np.diff(group_offsets)
            if group_sizes is not None and not np.array_equal(group_sizes, offset_sizes):
                raise ValueError("Ordered group_sizes and group_offsets must describe the same groups.")
            group_sizes = offset_sizes
        if ranking and group_sizes is None:
            raise ValueError("Ordered ranking requires explicit group_offsets or group_sizes.")
        scalars = _finite("training scalars", [learning_rate, l2_leaf_reg, bias, fold_len_multiplier], (4,))
        if not 0 < scalars[0] <= 1 or scalars[1] < 0 or (scalars[3] <= 1 and
                (group_sizes is None or float(fold_len_multiplier) <= 1)):
            raise ValueError("Invalid Ordered learning rate, L2 or fold growth.")
        initial = np.full(rows, scalars[2], np.float32) if initial_predictions is None else _finite("initial_predictions", initial_predictions, (rows,))
        group_offsets = None
        if group_sizes is not None:
            group_sizes = np.asarray(group_sizes)
            if (group_sizes.ndim != 1 or not 4 <= group_sizes.size <= rows or group_sizes.dtype.kind not in "iu"
                    or (group_sizes <= 0).any() or (group_sizes > rows).any()
                    or int(group_sizes.sum(dtype=np.uint64)) != rows):
                raise ValueError("Ordered requires at least four positive integer group sizes covering all rows.")
            if ranking or (group_sizes != 1).any():
                group_offsets = np.r_[np.uint32(0), np.cumsum(group_sizes, dtype=np.uint32)]
        if paired:
            from ._query_data import prepare_pair_arrays
            pair_winners, pair_losers, pair_weights = prepare_pair_arrays(
                pair_winners, pair_losers, pair_weights, rows, group_offsets)
            mass = np.bincount(pair_winners, weights=pair_weights, minlength=rows)
            mass += np.bincount(pair_losers, weights=pair_weights, minlength=rows)
            weights = np.ascontiguousarray(mass, np.float32)
        if queried:
            from ._query_data import query_metric
            query_metric(initial, targets, weights, group_offsets, objective, *map(float, query_parameters))
        if yeti:
            from ._query_data import validate_subgroup_hashes
            if (group_sizes > 1023).any():
                raise ValueError("Ordered YetiRank supports at most 1023 rows per query.")
            if not (group_sizes > 1).any():
                raise ValueError("YetiRank requires at least one query containing multiple rows.")
            self._targets, self._weights, self._offsets = targets.copy(), weights.copy(), group_offsets.copy()
            self._subgroups = validate_subgroup_hashes(subgroup_hashes, rows)
            if self._subgroups is not None:
                self._subgroups = self._subgroups.copy()
            self._legacy_centering = bool(legacy_prefix_centering)
        if permutations is None:
            permutations = np.stack([cuda_ordered_group_history_order(group_sizes, index, block_size)
                if group_offsets is not None else cuda_ordered_history_order(rows, index, block_size)
                for index in range(permutation_count)])
        else:
            permutations = np.asarray(permutations)
            if permutations.shape != (permutation_count, rows) or permutations.dtype.kind not in "iu":
                raise ValueError("permutations must be an integer matrix with shape [permutation_count,rows].")
            if (permutations < 0).any() or (permutations >= rows).any():
                raise ValueError("Invalid Ordered permutation row index.")
            if any(not np.all(np.bincount(order, minlength=rows) == 1) for order in permutations):
                raise ValueError("Each permutation must contain every row exactly once.")
        permutations = np.ascontiguousarray(permutations, np.uint32)
        if group_offsets is not None:
            for order in permutations:
                position = 0
                while position < rows:
                    first = int(order[position]); group = int(np.searchsorted(group_offsets[:-1], first))
                    if group >= len(group_sizes) or group_offsets[group] != first:
                        raise ValueError("Ordered permutations must preserve whole groups and their row order.")
                    size = int(group_sizes[group])
                    if position + size > rows or not np.array_equal(order[position:position + size], np.arange(first, first + size)):
                        raise ValueError("Ordered permutations must preserve whole groups and their row order.")
                    position += size
        self._params = Params(rows, features, len(cf), iterations, depth, objectives[objective],
            int(score_function == "NewtonCosine"), ("Newton", "Gradient", "Exact").index(leaf_estimation_method), leaf_steps, permutation_count,
            minimum, int(fold_size_loss_normalization), *map(float, scalars), parameter, 0, 0, 0)
        from ._training import _prepare_feature_penalties
        ctr_unique_values, feature_weights = _prepare_feature_penalties(bins, ctr_unique_values, model_size_reg, feature_weights)
        self._has_penalties = ctr_unique_values is not None
        if self._has_penalties:
            feature_weights = np.ones(features, np.float32) if feature_weights is None else feature_weights
        self.objective = objective
        digest = hashlib.sha256()
        for array in (bins, targets, weights, cf, cb, permutations):
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        if permutation_bins is not None:
            digest.update(b"ordered_permutation_feature_banks_v1")
            digest.update(permutation_bins.tobytes())
        # All-zero flags preserve the original numeric snapshot fingerprint.
        if candidate_types.any():
            digest.update(b"ordered_one_hot_candidates_v1")
            digest.update(candidate_types.tobytes())
        if group_offsets is not None:
            digest.update(b"ordered_group_offsets_v1")
            digest.update(group_offsets.tobytes())
            digest.update(np.float64(fold_len_multiplier).tobytes())
        if paired:
            digest.update(b"ordered_supplied_pairs_v1")
            for array in (pair_winners, pair_losers, pair_weights):
                digest.update(str(array.shape).encode())
                digest.update(array.tobytes())
        if yeti and self._subgroups is not None:
            digest.update(b"ordered_yeti_subgroups_v1")
            digest.update(self._subgroups.tobytes())
        if self._has_penalties:
            digest.update(b"ordered_static_ctr_penalties_v1")
            digest.update(ctr_unique_values.tobytes()); digest.update(feature_weights.tobytes())
            digest.update(np.float32(model_size_reg).tobytes())
        options = {name: getattr(self._params, name) for name, _ in Params._fields_ if name != "iterations"}
        options["random_seed"] = self._random_seed
        options["leaf_estimation_backtracking"] = leaf_estimation_backtracking
        options["selection_rng_policy"] = "numeric_ordered_host_v1"
        if queried:
            options.update(query_beta=float(query_parameters[0]), query_lambda=float(query_parameters[1]))
        if yeti:
            options.update(selection_rng_policy="ordered_yeti_host_v1", yeti_permutations=yeti_permutations,
                           decay=decay, legacy_prefix_centering=bool(legacy_prefix_centering),
                           simple_ctr_permutation_dependent=bool(simple_ctr_permutation_dependent))
        options["fold_permutation_block"] = block_size
        options.update(bootstrap_type=bootstrap_type, bagging_temperature=float(sampling[0]), subsample=float(sampling[1]),
                       random_strength=float(sampling[2]), observations_to_bootstrap=observations_to_bootstrap, mvs_reg=regularization)
        digest.update(json.dumps(options, sort_keys=True).encode())
        self._fingerprint = digest.hexdigest()
        if initial_state is not None and (not isinstance(initial_state, dict) or initial_state.get("version") != 1 or
                                          initial_state.get("fingerprint") != self._fingerprint):
            raise ValueError("Ordered state does not match the dataset, permutations or training options.")
        selection_state = None
        if initial_state is not None:
            offset = _integer("Ordered state iteration_offset", initial_state.get("iteration_offset"), 0, 2**32 - 1 - iterations)
            if self._iteration_offset not in (0, offset):
                raise ValueError("iteration_offset conflicts with Ordered state.")
            self._iteration_offset = offset
            selection_state = initial_state.get("selection_rng")
            if selection_state is None:
                raise ValueError("Ordered state requires persistent selection_rng state.")
        if yeti:
            from ._ordered_yeti_rng import OrderedYetiRankRng
            self._selection_rng = OrderedYetiRankRng(self._random_seed, permutation_count, leaf_steps,
                score_sets=1 + int(permutation_count > 1 and simple_ctr_permutation_dependent),
                initial_state=selection_state, iteration_offset=self._iteration_offset)
        else:
            self._selection_rng = OrderedSelectionRng(self._random_seed, permutation_count, selection_state,
                                                     iteration_offset=self._iteration_offset)
        self._lib = _load(build_library())
        error = ct.create_string_buffer(4096)
        if ranking:
            banks = bins if permutation_bins is None else permutation_bins
            arguments = [ct.byref(self._params), 1 if permutation_bins is None else permutation_count,
                         banks.size, _u8(banks)]
            if not paired:
                arguments += [_f32(targets), _f32(weights)]
            arguments += [_f32(initial), _u32(cf), _u32(cb), _u8(candidate_types), _u32(permutations)]
            if paired:
                objective_options = PairOptions(len(pair_winners), len(group_sizes), 0, 0)
                arguments += [ct.byref(objective_options), _u32(pair_winners), _u32(pair_losers), _f32(pair_weights)]
                operation = self._lib.cbm_ordered_session_create_pair_banked
            elif queried:
                objective_options = QueryOptions(len(group_sizes), *map(float, query_parameters), 0)
                arguments += [ct.byref(objective_options)]
                operation = self._lib.cbm_ordered_session_create_query_banked
            else:
                objective_options = YetiRankOptions(len(group_sizes), yeti_permutations, decay, int(legacy_prefix_centering))
                arguments += [ct.byref(objective_options)]
                operation = self._lib.cbm_ordered_session_create_yeti_banked
            arguments += [_u32(group_offsets), float(fold_len_multiplier), ct.byref(self._handle), error, len(error)]
            _check(operation(*arguments), error)
        elif permutation_bins is not None:
            _check(self._lib.cbm_ordered_session_create_banked(ct.byref(self._params), permutation_count, permutation_bins.size,
                _u8(permutation_bins), _f32(targets), _f32(weights), _f32(initial), _u32(cf), _u32(cb), _u8(candidate_types),
                _u32(permutations), len(group_sizes) if group_offsets is not None else 0,
                _u32(group_offsets) if group_offsets is not None else None, float(fold_len_multiplier),
                ct.byref(self._handle), error, len(error)), error)
        elif group_offsets is not None:
            _check(self._lib.cbm_ordered_session_create_grouped(ct.byref(self._params), _u8(bins), _f32(targets), _f32(weights),
                _f32(initial), _u32(cf), _u32(cb), _u8(candidate_types), _u32(permutations), len(group_sizes), _u32(group_offsets), float(fold_len_multiplier),
                ct.byref(self._handle), error, len(error)), error)
        elif candidate_types.any():
            _check(self._lib.cbm_ordered_session_create_typed(ct.byref(self._params), _u8(bins), _f32(targets), _f32(weights),
                _f32(initial), _u32(cf), _u32(cb), _u8(candidate_types), _u32(permutations), ct.byref(self._handle), error, len(error)), error)
        else:
            _check(self._lib.cbm_ordered_session_create(ct.byref(self._params), _u8(bins), _f32(targets), _f32(weights),
                _f32(initial), _u32(cf), _u32(cb), _u32(permutations), ct.byref(self._handle), error, len(error)), error)
        try:
            tasks, count = ct.c_uint32(), ct.c_uint32()
            _check(self._lib.cbm_ordered_session_state_shape(self._handle, ct.byref(tasks), ct.byref(count), error, len(error)), error)
            self._task_count, self._cursor_count = tasks.value, count.value
            if initial_state is not None:
                if not isinstance(initial_state, dict) or initial_state.get("version") != 1 or initial_state.get("fingerprint") != self._fingerprint:
                    raise ValueError("Ordered state does not match the dataset, permutations or training options.")
                restored = _finite("Ordered state cursors", initial_state.get("cursors"), (self._cursor_count,))
                offset = _integer("Ordered state iteration_offset", initial_state.get("iteration_offset"), 0, 2**32 - 1 - iterations)
                if self._iteration_offset not in (0, offset):
                    raise ValueError("iteration_offset conflicts with Ordered state.")
                descriptors, _ = self._copy_state()
                if not np.array_equal(initial_state.get("descriptors"), descriptors):
                    raise ValueError("Ordered state fold descriptors do not match.")
                _check(self._lib.cbm_ordered_session_restore_cursors(self._handle, self._cursor_count,
                    _f32(restored), error, len(error)), error)
                self._iteration_offset = offset
                saved_lambda = initial_state.get("mvs_lambda")
                if saved_lambda is not None:
                    saved_lambda = float(_finite("Ordered state mvs_lambda", saved_lambda, ()))
                    if saved_lambda < 0 or bootstrap_type != "MVS":
                        raise ValueError("Invalid Ordered state MVS regularization.")
                if initial_lambda is not None and initial_lambda != saved_lambda:
                    raise ValueError("initial_mvs_lambda conflicts with Ordered state.")
                if bootstrap_type == "MVS" and regularization is None and offset and saved_lambda is None:
                    raise ValueError("Ordered adaptive MVS resume requires saved lambda state.")
                initial_lambda = saved_lambda
            bootstrap = BootstrapOptions(("No", "Bayesian", "Bernoulli", "Poisson", "MVS").index(bootstrap_type),
                self._random_seed & 0xffffffff, self._random_seed >> 32, self._iteration_offset,
                float(sampling[0]), float(sampling[1]), regularization or 0, regularization is not None,
                initial_lambda or 0, initial_lambda is not None, 0, 0)
            noise = ScoreNoiseOptions(float(sampling[2]), 0, 0, 0)
            _check(self._lib.cbm_ordered_session_set_bootstrap(self._handle, ct.byref(bootstrap),
                int(observations_to_bootstrap == "TestOnly"), error, len(error)), error)
            _check(self._lib.cbm_ordered_session_set_score_noise(self._handle, ct.byref(noise), error, len(error)), error)
            _check(self._lib.cbm_ordered_session_set_backtracking(self._handle,
                ("No", "AnyImprovement", "Armijo").index(leaf_estimation_backtracking), error, len(error)), error)
            if self._has_penalties:
                penalty = FeaturePenaltyOptions(float(model_size_reg), 0, 0, 0)
                _check(self._lib.cbm_ordered_session_set_feature_penalties(self._handle, ct.byref(penalty),
                    _u32(ctr_unique_values), _f32(feature_weights), error, len(error)), error)
            self._initial_loss = self._metric() if yeti else self._info().loss
        except Exception:
            self.close()
            raise

    def _require_open(self):
        if not self._handle.value:
            raise RuntimeError("Ordered training session is closed.")

    def _info(self):
        self._require_open()
        info, error = StepInfo(), ct.create_string_buffer(4096)
        _check(self._lib.cbm_ordered_session_info(self._handle, ct.byref(info), error, len(error)), error)
        return info

    def _metric(self):
        from ._training import _shared_metric
        return _shared_metric("PFound", self.predictions(), self._targets, self._weights,
                              self._offsets, subgroup_hashes=self._subgroups)

    def _set_yeti_seeds(self, seeds, *, leaves=False):
        values = np.ascontiguousarray(seeds, np.uint64)
        error = ct.create_string_buffer(4096)
        operation = (self._lib.cbm_ordered_session_set_yeti_leaf_seeds if leaves
                     else self._lib.cbm_ordered_session_set_yeti_oracle_seeds)
        _check(operation(self._handle, len(values), values.ctypes.data_as(ct.POINTER(ct.c_uint64)),
                         error, len(error)), error)

    @property
    def completed_iterations(self):
        return len(self._steps)

    @property
    def closed(self):
        return not bool(self._handle.value)

    @property
    def bootstrap_state(self):
        with self._lock:
            self._require_open()
            iteration, value, valid = ct.c_uint32(), ct.c_float(), ct.c_uint32()
            error = ct.create_string_buffer(4096)
            _check(self._lib.cbm_ordered_session_get_bootstrap_state(self._handle, ct.byref(iteration), ct.byref(value),
                ct.byref(valid), error, len(error)), error)
            return {"iteration_offset": iteration.value, "mvs_lambda": float(value.value) if valid.value else None}

    def predictions(self):
        with self._lock:
            self._require_open()
            values, error = np.empty(self._params.rows, np.float32), ct.create_string_buffer(4096)
            _check(self._lib.cbm_ordered_session_copy_predictions(self._handle, _f32(values), error, len(error)), error)
            return values

    def _copy_state(self):
        descriptors = np.empty((self._task_count, 4), np.uint32)
        cursors, error = np.empty(self._cursor_count, np.float32), ct.create_string_buffer(4096)
        _check(self._lib.cbm_ordered_session_copy_state(self._handle, self._task_count, self._cursor_count,
            _u32(descriptors), _f32(cursors), error, len(error)), error)
        return descriptors, cursors

    def state(self):
        with self._lock:
            self._require_open()
            descriptors, cursors = self._copy_state()
            return {"version": 1, "fingerprint": self._fingerprint, "descriptors": descriptors,
                    "cursors": cursors, "selection_rng": self._selection_rng.state(), **self.bootstrap_state}

    def step(self):
        with self._lock:
            self._require_open()
            p = self._params
            if self.completed_iterations >= p.iterations:
                raise RuntimeError("Ordered session has no remaining iterations.")
            features, borders, types = np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint8)
            values, weights = np.zeros(1 << p.depth, np.float32), np.zeros(1 << p.depth, np.float32)
            info, depth, error = StepInfo(), ct.c_uint32(), ct.create_string_buffer(4096)
            selected = self._selection_rng.select()
            if self._yeti:
                weak_count, leaf_count = ct.c_uint32(), ct.c_uint32()
                _check(self._lib.cbm_ordered_session_yeti_seed_shape(self._handle, selected,
                    ct.byref(weak_count), ct.byref(leaf_count), error, len(error)), error)
                self._set_yeti_seeds(self._selection_rng.weak(weak_count.value))
                _check(self._lib.cbm_ordered_session_begin_tree(self._handle, selected, error, len(error)), error)
                attempts = 0
                while True:
                    structure = StructureInfo()
                    _check(self._lib.cbm_ordered_session_grow_tree(self._handle, ct.byref(structure), error, len(error)), error)
                    if p.candidates and p.depth:
                        attempts += 1
                    if structure.finished:
                        break
                self._set_yeti_seeds(self._selection_rng.leaves(attempts, leaf_count.value), leaves=True)
                _check(self._lib.cbm_ordered_session_finish_tree(self._handle, ct.byref(info), ct.byref(depth),
                    _u32(features), _u32(borders), _u8(types), _f32(values), _f32(weights), error, len(error)), error)
                self._selection_rng.finish()
                loss = self._metric()
            else:
                _check(self._lib.cbm_ordered_session_step(self._handle, selected, ct.byref(info), ct.byref(depth),
                    _u32(features), _u32(borders), _u8(types), _f32(values), _f32(weights), error, len(error)), error)
                loss = float(info.loss)
            count = depth.value
            # The numeric native loop only exits early for an invalid/repeated
            # winner. Its final unsuccessful attempt still consumes a CUDA host
            # score seed; empty candidates and depth zero never enter that loop.
            if not self._yeti:
                attempts = 0 if not p.candidates or not p.depth else min(count + 1, p.depth)
                self._selection_rng.finish(attempts)
            result = StepResult(info.completed_iterations, bool(info.finished), count, features[:count].copy(),
                borders[:count].copy(), types[:count].copy(), values[:1 << count].copy(), weights[:1 << count].copy(),
                loss, _stats(info.stats))
            result.stats.update(boosting_type="Ordered", search_permutation=selected)
            self._steps.append(result)
            return result

    def result(self):
        with self._lock:
            self._require_open()
            p, count = self._params, self.completed_iterations
            if count * (1 << p.depth) * 8 > 512 * 1024**2:
                raise ValueError("Padded Ordered output exceeds 512 MiB; consume incremental steps.")
            result = TrainResult(np.zeros(count, np.uint32), np.zeros((count, p.depth), np.uint32),
                np.zeros((count, p.depth), np.uint32), np.zeros((count, 1 << p.depth), np.float32),
                np.zeros((count, 1 << p.depth), np.float32), self.predictions(),
                np.array([self._initial_loss] + [step.loss for step in self._steps], np.float32),
                _stats(self._info().stats), np.zeros((count, p.depth), np.uint8))
            for index, step in enumerate(self._steps):
                result.depths[index] = step.depth
                result.split_features[index, :step.depth] = step.split_features
                result.split_bins[index, :step.depth] = step.split_bins
                result.split_types[index, :step.depth] = step.split_types
                result.leaf_values[index, :1 << step.depth] = step.leaf_values
                result.leaf_weights[index, :1 << step.depth] = step.leaf_weights
            result.stats.update(boosting_type="Ordered", permutation_count=p.permutations,
                ordered_tasks=self._task_count, ordered_cursor_count=self._cursor_count,
                bootstrap_state=self.bootstrap_state,
                search_permutations=[step.stats["search_permutation"] for step in self._steps])
            if self._yeti:
                result.stats["yeti_centering"] = "legacy_prefix" if self._legacy_centering else "all_rows"
            return result

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if self._handle.value and self._lib is not None:
                    self._lib.cbm_ordered_session_close(self._handle)
                    self._handle = ct.c_void_p()

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def train(bins, targets, candidate_features, candidate_bins, **kwargs):
    with Session(bins, targets, candidate_features, candidate_bins, **kwargs) as session:
        for _ in range(session._params.iterations):
            session.step()
        return session.result()
