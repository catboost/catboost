"""Validated persistent Metal training sessions; no CPU/CUDA training fallback."""
import ctypes as ct
from dataclasses import dataclass
import functools
import hashlib
import numbers
from pathlib import Path
import platform
import subprocess
import threading

import numpy as np


class TrainParams(ct.Structure):
    _fields_ = [
        ("rows", ct.c_uint32), ("features", ct.c_uint32),
        ("candidates", ct.c_uint32), ("bins_per_feature", ct.c_uint32),
        ("iterations", ct.c_uint32), ("depth", ct.c_uint32),
        ("score_function", ct.c_uint32), ("learning_rate", ct.c_float),
        ("l2_leaf_reg", ct.c_float), ("bias", ct.c_float),
    ]


class SessionParams(ct.Structure):
    _fields_ = [("train", TrainParams), ("objective", ct.c_uint32),
                ("leaf_estimation_iterations", ct.c_uint32),
                ("leaf_estimation_backtracking", ct.c_uint32), ("reserved", ct.c_uint32)]


class TrainStats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("device_name", ct.c_char * 256)]


class StepInfo(ct.Structure):
    _fields_ = [("completed_iterations", ct.c_uint32), ("finished", ct.c_uint32),
                ("loss", ct.c_float), ("reserved", ct.c_uint32), ("stats", TrainStats)]


class StructureInfo(ct.Structure):
    _fields_ = [("depth", ct.c_uint32), ("finished", ct.c_uint32), ("has_split", ct.c_uint32),
                ("feature", ct.c_uint32), ("bin", ct.c_uint32), ("type", ct.c_uint32),
                ("score", ct.c_float), ("gain", ct.c_float)]


class AppendFeatureOptions(ct.Structure):
    _fields_ = [("permutation_count", ct.c_uint32), ("features", ct.c_uint32),
                ("candidates", ct.c_uint32), ("bins_per_feature", ct.c_uint32),
                ("reserved", ct.c_uint32 * 4)]


class ObjectiveOptions(ct.Structure):
    _fields_ = [("objective", ct.c_uint32), ("leaf_estimation_method", ct.c_uint32),
                ("objective_param", ct.c_float), ("reserved", ct.c_uint32)]


class QueryOptions(ct.Structure):
    _fields_ = [("group_count", ct.c_uint32), ("beta", ct.c_float),
                ("lambda_", ct.c_float), ("reserved", ct.c_uint32)]


class PairOptions(ct.Structure):
    _fields_ = [("pair_count", ct.c_uint32), ("group_count", ct.c_uint32),
                ("reserved0", ct.c_uint32), ("reserved1", ct.c_uint32)]


class YetiRankOptions(ct.Structure):
    _fields_ = [("group_count", ct.c_uint32), ("permutations", ct.c_uint32),
                ("decay", ct.c_float), ("legacy_prefix_centering", ct.c_uint32)]


class BootstrapOptions(ct.Structure):
    _fields_ = [("bootstrap_type", ct.c_uint32), ("random_seed_low", ct.c_uint32),
                ("random_seed_high", ct.c_uint32), ("iteration_offset", ct.c_uint32),
                ("bagging_temperature", ct.c_float), ("subsample", ct.c_float),
                ("mvs_reg", ct.c_float), ("mvs_reg_is_set", ct.c_uint32),
                ("initial_mvs_lambda", ct.c_float), ("initial_mvs_lambda_is_set", ct.c_uint32),
                ("reserved0", ct.c_uint32), ("reserved1", ct.c_uint32)]


class FeaturePenaltyOptions(ct.Structure):
    _fields_ = [("model_size_reg", ct.c_float), ("reserved0", ct.c_uint32),
                ("reserved1", ct.c_uint32), ("reserved2", ct.c_uint32)]


class ScoreNoiseOptions(ct.Structure):
    _fields_ = [("random_strength", ct.c_float), ("reserved0", ct.c_uint32),
                ("reserved1", ct.c_uint32), ("reserved2", ct.c_uint32)]


@dataclass
class TrainResult:
    depths: np.ndarray
    split_features: np.ndarray
    split_bins: np.ndarray
    leaf_values: np.ndarray
    leaf_weights: np.ndarray
    predictions: np.ndarray
    rmse: np.ndarray
    stats: dict
    split_types: np.ndarray | None = None

    @property
    def loss(self):
        """Objective loss history; rmse is retained as its legacy field name."""
        return self.rmse

    @property
    def completed_iterations(self):
        return len(self.depths)


@dataclass
class StepResult:
    completed_iterations: int
    finished: bool
    depth: int
    split_features: np.ndarray
    split_bins: np.ndarray
    split_types: np.ndarray
    leaf_values: np.ndarray
    leaf_weights: np.ndarray
    loss: float
    stats: dict

    @property
    def iteration(self):
        return self.completed_iterations - 1


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("The Metal port requires macOS on Apple Silicon.")
    import fcntl
    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    sources = [native / "metal_trainer.mm", native / "metal_sort.mm", native / "metal_exception.cpp",
               *sorted(native.glob("*.h"))]
    digest = hashlib.sha256(platform.platform().encode())
    for source in sources:
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_{digest.hexdigest()[:20]}.dylib"
    with (build / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            command = ["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                       "-framework", "Foundation", "-framework", "Metal",
                       str(native / "metal_trainer.mm"), str(native / "metal_sort.mm"),
                       str(native / "metal_exception.cpp"), "-o", str(temporary)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError("Could not build Metal runtime:\n" + result.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    lib = ct.CDLL(str(path))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    text = [ct.c_char_p, ct.c_size_t]
    lib.cbm_device_info.argtypes = text + text
    lib.cbm_device_info.restype = ct.c_int
    lib.cbm_train.argtypes = [ct.POINTER(TrainParams), u8, f32, u32, u32,
                             u32, u32, u32, f32, f32, f32, f32,
                             ct.POINTER(TrainStats)] + text
    lib.cbm_train.restype = ct.c_int
    lib.cbm_session_create.argtypes = [ct.POINTER(SessionParams), u8, f32, f32, f32,
                                      u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_session_create_configured.argtypes = [ct.POINTER(SessionParams), ct.POINTER(ObjectiveOptions),
        u8, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_session_create_query.argtypes = [ct.POINTER(SessionParams), ct.POINTER(ObjectiveOptions),
        ct.POINTER(QueryOptions), u32, u8, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_session_create_pair.argtypes = [ct.POINTER(SessionParams), ct.POINTER(ObjectiveOptions),
        ct.POINTER(PairOptions), u32, u32, f32, u32, u8, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_session_step.argtypes = [ct.c_void_p, ct.POINTER(StepInfo), u32, u32, u32, u8,
                                    f32, f32] + text
    lib.cbm_session_begin_tree.argtypes = [ct.c_void_p] + text
    lib.cbm_session_grow_tree.argtypes = [ct.c_void_p, ct.POINTER(StructureInfo)] + text
    lib.cbm_session_finish_tree.argtypes = lib.cbm_session_step.argtypes
    lib.cbm_session_append_features.argtypes = [ct.c_void_p, ct.POINTER(AppendFeatureOptions), ct.POINTER(u8),
        u32, u32, u8, u32, f32, u8, u8, u32] + text
    lib.cbm_session_set_feature_activity.argtypes = [ct.c_void_p, ct.c_uint32, u8] + text
    lib.cbm_session_copy_feature_metadata.argtypes = [ct.c_void_p, ct.c_uint32, u32, f32, u8, u8, u8] + text
    lib.cbm_session_result.argtypes = [ct.c_void_p, ct.c_uint32, u32, u32, u32, u32, u8,
                                      f32, f32, f32, f32, ct.POINTER(TrainStats)] + text
    for name in ("cbm_session_create", "cbm_session_create_configured", "cbm_session_create_query",
                 "cbm_session_create_pair", "cbm_session_step", "cbm_session_result",
                 "cbm_session_begin_tree", "cbm_session_grow_tree", "cbm_session_finish_tree",
                 "cbm_session_append_features", "cbm_session_set_feature_activity", "cbm_session_copy_feature_metadata"):
        getattr(lib, name).restype = ct.c_int
    lib.cbm_session_close.argtypes = [ct.c_void_p]
    lib.cbm_session_close.restype = None
    lib.cbm_session_set_objective.argtypes = [ct.c_void_p, ct.POINTER(ObjectiveOptions)] + text
    lib.cbm_session_set_bootstrap.argtypes = [ct.c_void_p, ct.POINTER(BootstrapOptions)] + text
    lib.cbm_session_set_score_noise.argtypes = [ct.c_void_p, ct.POINTER(ScoreNoiseOptions)] + text
    lib.cbm_session_get_bootstrap_state.argtypes = [ct.c_void_p, u32, f32, u32] + text
    lib.cbm_session_set_permutations.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(u8),
                                                ct.POINTER(f32), f32, u8] + text
    lib.cbm_session_select_permutation.argtypes = [ct.c_void_p, ct.c_uint32] + text
    lib.cbm_session_copy_permutation_state.argtypes = [ct.c_void_p, ct.c_uint32, f32, f32, u8] + text
    lib.cbm_session_set_feature_penalties.argtypes = [ct.c_void_p, ct.POINTER(FeaturePenaltyOptions), u32, f32, u8] + text
    lib.cbm_session_copy_feature_penalty_state.argtypes = [ct.c_void_p, u8] + text
    lib.cbm_session_get_workspace_info.argtypes = [ct.c_void_p, u32, ct.POINTER(ct.c_uint64), ct.POINTER(ct.c_uint64)] + text
    lib.cbm_session_copy_predictions.argtypes = [ct.c_void_p, f32] + text
    for name in ("cbm_session_set_objective", "cbm_session_set_bootstrap",
                 "cbm_session_get_bootstrap_state", "cbm_session_copy_predictions", "cbm_session_set_score_noise",
                 "cbm_session_set_permutations", "cbm_session_select_permutation", "cbm_session_copy_permutation_state",
                 "cbm_session_get_workspace_info", "cbm_session_set_feature_penalties", "cbm_session_copy_feature_penalty_state"):
        getattr(lib, name).restype = ct.c_int
    return lib


def _u8(value):
    return None if value is None else value.ctypes.data_as(ct.POINTER(ct.c_uint8))


def _u32(value):
    return None if value is None else value.ctypes.data_as(ct.POINTER(ct.c_uint32))


def _f32(value):
    return None if value is None else value.ctypes.data_as(ct.POINTER(ct.c_float))


def _stats(value):
    return {"device": value.device_name.decode("utf-8"),
            "kernel_dispatches": int(value.kernel_dispatches), "gpu_seconds": float(value.gpu_seconds)}


def device_info():
    lib = _load(build_library())
    name, error = ct.create_string_buffer(256), ct.create_string_buffer(2048)
    if lib.cbm_device_info(name, len(name), error, len(error)):
        raise RuntimeError(error.value.decode("utf-8", errors="replace"))
    return {"name": name.value.decode("utf-8"), "backend": "Metal"}


def _prepare(bins, targets, candidate_features, candidate_bins, *, iterations, depth,
             learning_rate, l2_leaf_reg, bias, score_function, objective, sample_weight,
             leaf_estimation_iterations, leaf_estimation_backtracking, initial_predictions,
             candidate_types, objective_param, leaf_estimation_method, bootstrap_type, random_seed,
             iteration_offset, bagging_temperature, subsample, mvs_reg, initial_mvs_lambda, random_strength,
             group_offsets=None, query_beta=1.0, query_lambda=0.01,
             pair_winners=None, pair_losers=None, pair_weights=None, _full_matrix=False):
    bins, targets = np.asarray(bins), np.asarray(targets)
    candidate_features, candidate_bins = np.asarray(candidate_features), np.asarray(candidate_bins)
    if bins.ndim != 2 or bins.dtype.kind not in "iu" or not all(bins.shape):
        raise ValueError("bins must be a nonempty two-dimensional integer array.")
    features, rows = bins.shape
    if rows > 16777216 or rows * features > np.iinfo(np.uint32).max:
        raise ValueError("Quantized data exceeds the Metal row/index limit.")
    if (bins < 0).any() or (bins > 255).any():
        raise ValueError("Quantized bins must be in [0, 255].")
    if targets.shape != (rows,) or targets.dtype.kind not in "biuf":
        raise ValueError("targets must be a numeric vector with one entry per row.")
    if (candidate_features.ndim != 1 or candidate_bins.ndim != 1
            or candidate_features.shape != candidate_bins.shape):
        raise ValueError("Candidate feature and border arrays must be matching vectors.")
    if candidate_features.size and (
            candidate_features.dtype.kind not in "iu" or candidate_bins.dtype.kind not in "iu"
            or (candidate_features < 0).any() or (candidate_features >= features).any()
            or (candidate_bins < 0).any() or (candidate_bins >= 255).any()):
        raise ValueError("Candidate feature/border indices are invalid.")
    if candidate_types is None:
        candidate_types = np.zeros(candidate_features.size, np.uint8)
    else:
        candidate_types = np.asarray(candidate_types)
        if (candidate_types.shape != candidate_features.shape or candidate_types.dtype.kind not in "biu"
                or ((candidate_types != 0) & (candidate_types != 1)).any()):
            raise ValueError("candidate_types must be a matching vector of numeric(0)/one-hot(1) flags.")
        candidate_types = np.ascontiguousarray(candidate_types, dtype=np.uint8)
    for name, value, minimum, maximum in (("iterations", iterations, 1, 10000),
                                          ("depth", depth, 0, 16),
                                          ("leaf_estimation_iterations", leaf_estimation_iterations, 1, 1000)):
        if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
                or not minimum <= value <= maximum):
            raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    if score_function not in ("L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2", "SatL2"):
        raise ValueError("Unsupported Metal split score function.")
    objective_ids = {"RMSE": 0, "Logloss": 1, "CrossEntropy": 2, "Poisson": 3, "Huber": 4, "Expectile": 5,
                     "Lq": 6, "Tweedie": 7, "LogLinQuantile": 8, "Quantile": 9, "MAE": 10, "MAPE": 11,
                     "QueryRMSE": 12, "QuerySoftMax": 13, "PairLogit": 14}
    if objective not in objective_ids:
        raise ValueError("Unsupported Metal objective.")
    grouped = objective in ("QueryRMSE", "QuerySoftMax")
    paired = objective == "PairLogit"
    if (grouped and group_offsets is None) or (group_offsets is not None and not (grouped or paired)):
        raise ValueError("group_offsets are required for query losses, optional for PairLogit, and unsupported otherwise.")
    if group_offsets is not None:
        group_offsets = np.asarray(group_offsets)
        if (group_offsets.ndim != 1 or group_offsets.dtype.kind not in "iu"
                or not 2 <= group_offsets.size <= rows + 1 or group_offsets[0] != 0
                or group_offsets[-1] != rows or (group_offsets[1:] <= group_offsets[:-1]).any()):
            raise ValueError("group_offsets must increase strictly from zero through all training rows.")
        group_offsets = np.ascontiguousarray(group_offsets, dtype=np.uint32)
    if paired:
        if sample_weight is not None:
            raise ValueError("PairLogit uses incident pair mass; sample_weight must be None.")
        pair_winners, pair_losers = np.asarray(pair_winners), np.asarray(pair_losers)
        if (pair_winners.ndim != 1 or pair_winners.dtype.kind not in "iu"
                or pair_losers.shape != pair_winners.shape or pair_losers.dtype.kind not in "iu"
                or not 0 < pair_winners.size <= np.iinfo(np.uint32).max // 2
                or (pair_winners < 0).any() or (pair_winners >= rows).any()
                or (pair_losers < 0).any() or (pair_losers >= rows).any()
                or (pair_winners == pair_losers).any()):
            raise ValueError("PairLogit requires matching nonempty valid winner/loser vectors without self-pairs.")
        pair_winners = np.ascontiguousarray(pair_winners, dtype=np.uint32)
        pair_losers = np.ascontiguousarray(pair_losers, dtype=np.uint32)
        if pair_weights is None:
            pair_weights = np.ones(pair_winners.size, np.float32)
        else:
            pair_weights = np.asarray(pair_weights)
            if pair_weights.shape != pair_winners.shape or pair_weights.dtype.kind not in "biuf":
                raise ValueError("pair_weights must be a numeric vector matching supplied pairs.")
            with np.errstate(over="ignore", invalid="ignore"):
                pair_weights = np.ascontiguousarray(pair_weights, dtype=np.float32)
        pair_mass = pair_weights.sum(dtype=np.float64)
        if (not np.isfinite(pair_weights).all() or (pair_weights < 0).any()
                or not 0 < 2 * pair_mass <= np.finfo(np.float32).max):
            raise ValueError("PairLogit pair weights need positive finite float32 incident mass.")
        if group_offsets is not None:
            winner_groups = np.searchsorted(group_offsets, pair_winners, side="right")
            loser_groups = np.searchsorted(group_offsets, pair_losers, side="right")
            if (winner_groups != loser_groups).any():
                raise ValueError("PairLogit endpoints must belong to the same query group.")
    elif pair_winners is not None or pair_losers is not None or pair_weights is not None:
        raise ValueError("Supplied pair arrays require the PairLogit objective.")
    with np.errstate(over="ignore", invalid="ignore"):
        query_beta, query_lambda = np.float32(query_beta), np.float32(query_lambda)
    if not np.isfinite([query_beta, query_lambda]).all():
        raise ValueError("query_beta and query_lambda must be finite float32 values.")
    if leaf_estimation_method not in ("Newton", "Gradient", "Exact", "Simple"):
        raise ValueError("leaf_estimation_method must be Newton, Gradient, Exact or Simple.")
    if leaf_estimation_method == "Simple" and (not _full_matrix or leaf_estimation_iterations != 1
                                               or depth == 0 or candidate_features.size == 0):
        raise ValueError("Simple leaves require a full-matrix target, one estimation iteration and depth 1..8 with split candidates.")
    if objective in ("Huber", "Expectile", "Lq", "Tweedie") and objective_param is None:
        raise ValueError(f"{objective} requires an explicit objective_param.")
    with np.errstate(over="ignore", invalid="ignore"):
        parameter = np.float32((0.5 if objective in ("LogLinQuantile", "Quantile", "MAE") else 1)
                               if objective_param is None else objective_param)
    if (not np.isfinite(parameter) or (objective == "Huber" and parameter < 0)
            or (objective == "Expectile" and not 0 <= parameter <= 1)):
        raise ValueError("Invalid objective_param: Huber delta must be nonnegative; Expectile alpha must be in [0, 1].")
    if ((objective == "Lq" and parameter < 1) or (objective == "Tweedie" and not 1 < parameter < 2)
            or (objective in ("LogLinQuantile", "Quantile") and not 0 <= parameter <= 1)):
        raise ValueError("Invalid objective_param for Lq, Tweedie or Quantile.")
    if leaf_estimation_method == "Newton" and (objective in ("LogLinQuantile", "Quantile", "MAE", "MAPE")
                                                or (objective == "Lq" and parameter < 2)):
        raise ValueError("Newton is unsupported for this objective or objective parameter.")
    if leaf_estimation_method == "Exact" and objective not in ("Quantile", "MAE", "MAPE"):
        raise ValueError("Exact supports Quantile, MAE and MAPE only.")
    if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
        raise ValueError("leaf_estimation_backtracking must be No, AnyImprovement or Armijo.")
    with np.errstate(over="ignore", invalid="ignore"):
        targets = np.ascontiguousarray(targets, dtype=np.float32)
        learning_rate, l2_leaf_reg, bias = map(np.float32, (learning_rate, l2_leaf_reg, bias))
    if (not np.isfinite(targets).all() or not np.isfinite([learning_rate, l2_leaf_reg, bias]).all()
            or not 0 < learning_rate <= 1 or l2_leaf_reg < 0):
        raise ValueError("Targets and training scalars must be valid finite float32 values.")
    if objective == "Logloss" and ((targets != 0) & (targets != 1)).any():
        raise ValueError("Logloss targets must be 0 or 1.")
    if objective == "CrossEntropy" and ((targets < 0) | (targets > 1)).any():
        raise ValueError("CrossEntropy targets must be in [0, 1].")
    if objective in ("Poisson", "Tweedie", "QuerySoftMax") and (targets < 0).any():
        raise ValueError("Poisson/Tweedie/QuerySoftMax targets must be nonnegative.")
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.shape != (rows,) or sample_weight.dtype.kind not in "biuf":
            raise ValueError("sample_weight must be a numeric vector with one entry per row.")
        with np.errstate(over="ignore", invalid="ignore"):
            sample_weight = np.ascontiguousarray(sample_weight, dtype=np.float32)
        total_weight = sample_weight.sum(dtype=np.float64)
        if (not np.isfinite(sample_weight).all() or (sample_weight < 0).any()
                or not 0 < total_weight <= np.finfo(np.float32).max):
            raise ValueError("Sample weights must be finite, nonnegative, with positive finite float32 total.")
    if objective == "QuerySoftMax":
        target_mass = np.sum(targets.astype(np.float64) * (1 if sample_weight is None else sample_weight), dtype=np.float64)
        if not 0 < target_mass <= np.finfo(np.float32).max:
            raise ValueError("QuerySoftMax requires positive finite float32 total weighted target mass.")
    if initial_predictions is not None:
        initial_predictions = np.asarray(initial_predictions)
        if initial_predictions.shape != (rows,) or initial_predictions.dtype.kind not in "biuf":
            raise ValueError("initial_predictions must be a numeric vector with one entry per row.")
        with np.errstate(over="ignore", invalid="ignore"):
            initial_predictions = np.ascontiguousarray(initial_predictions, dtype=np.float32)
        if not np.isfinite(initial_predictions).all():
            raise ValueError("Initial raw predictions must be finite float32 values.")
    bins = np.ascontiguousarray(bins, dtype=np.uint8)
    candidate_features = np.ascontiguousarray(candidate_features, dtype=np.uint32)
    candidate_bins = np.ascontiguousarray(candidate_bins, dtype=np.uint32)
    max_bins = int(bins.max()) + 1
    if candidate_bins.size:
        required = candidate_bins.astype(np.uint64) + np.where(candidate_types == 0, 2, 1)
        max_bins = max(max_bins, int(required.max()))
        feature_flags = {}
        for feature, flag in zip(candidate_features, candidate_types):
            prior = feature_flags.setdefault(int(feature), int(flag))
            if prior != flag:
                raise ValueError("A feature cannot mix numeric and one-hot candidate types.")
    leaf_count = 1 << depth
    if not isinstance(_full_matrix, bool) or (_full_matrix and depth > 8):
        raise ValueError("Full-matrix training supports depth <= 8 like CUDA.")
    output_bytes = iterations * (leaf_count * 8 + depth * 9 + 8) + rows * 4 + 4
    if depth <= 8 and output_bytes > 512 * 1024 * 1024:
        raise ValueError("Training output exceeds the experimental 512 MiB limit.")
    partition_tiles = (rows + 4095) // 4096
    histogram_tiles = min(partition_tiles, 32)
    score_groups = max(1, min((len(candidate_features) + 255) // 256, 64))
    compact_spans = np.zeros(features, np.uint32)
    if len(candidate_features):
        np.maximum.at(compact_spans, candidate_features, candidate_bins + 1)
    compact_bins = int(compact_spans.sum(dtype=np.uint64))
    histogram_cells = 1 if _full_matrix else leaf_count * (max(1, compact_bins) if depth > 8 else features * max_bins)
    # Keep in sync with native Session allocation accounting (compensated float4 pairs).
    working_bytes = (features * rows + rows * 36 + len(candidate_features) * 9 + features * 13
                     + histogram_cells * 8 + leaf_count * 24 + 4
                     + ((rows + 255) // 256 * 4 + (rows + 65535) // 65536 * 4 + (leaf_count + 1) * 4
                        if depth > 8 else leaf_count * partition_tiles * 4) + leaf_count * histogram_tiles * 32
                     + (score_groups + 1) * 32 + min((rows + 255) // 256, 4096) * 4)
    working_bytes += (features + 1) * 4
    if _full_matrix:
        # These targets own G/H and full leaf projections; shared scalar buffers
        # are minimal binding placeholders, as in native Session.
        working_bytes -= (rows - 1) * 8 + (leaf_count - 1) * 8 + (leaf_count * histogram_tiles - 1) * 32
    if depth > 8:
        working_bytes += ((rows + 8191) // 8192 + min(rows, leaf_count)) * 16 + leaf_count * 4 + 52
    if leaf_estimation_backtracking != "No" and leaf_estimation_iterations > 1:
        working_bytes += leaf_count * 16 + min((rows + 255) // 256, 4096) * 8
    bootstrap_ids = {"No": 0, "Bayesian": 1, "Bernoulli": 2, "Poisson": 3, "MVS": 4}
    if bootstrap_type not in bootstrap_ids:
        raise ValueError("Unsupported bootstrap_type.")
    for name, value, maximum in (("random_seed", random_seed, 2**64 - 1),
                                  ("iteration_offset", iteration_offset, 2**32 - 1 - iterations)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be a nonnegative integer no greater than {maximum}.")
    with np.errstate(over="ignore", invalid="ignore"):
        strength = np.float32(random_strength)
        temperature, fraction = np.float32(bagging_temperature), np.float32(subsample)
        regularization = np.float32(0 if mvs_reg is None else mvs_reg)
        initial_lambda = np.float32(0 if initial_mvs_lambda is None else initial_mvs_lambda)
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("random_strength must be finite and nonnegative.")
    if not _full_matrix and strength > 0 and score_function in ("Cosine", "NewtonCosine"):
        working_bytes += min((rows + 255) // 256, 4096) * 4
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError("bagging_temperature must be finite and nonnegative.")
    if not np.isfinite(fraction) or not 0 < fraction <= 1 or (bootstrap_type == "Poisson" and fraction >= 1):
        raise ValueError("subsample must be in (0, 1], and strictly below 1 for Poisson bootstrap.")
    if (not np.isfinite([regularization, initial_lambda]).all() or regularization < 0 or initial_lambda < 0):
        raise ValueError("mvs_reg and initial_mvs_lambda must be finite and nonnegative.")
    if not _full_matrix and bootstrap_type != "No":
        working_bytes += rows * 8 + min((rows + 255) // 256, 4096) * 8 + ((rows + 8191) // 8192) * 4
    if leaf_estimation_method == "Exact":
        working_bytes += rows * 50 + leaf_count * histogram_tiles * 8 + leaf_count * 12
    if grouped:
        working_bytes += rows * 4 + (group_offsets.size - 1) * 12 + 8 + min((rows + 255) // 256, 4096) * 8
    if paired:
        working_bytes += pair_winners.size * 44 + rows * 8 + 12 + min((rows + 255) // 256, 4096) * 8 + ((leaf_count + 255) // 256) * 8
    if depth > 8:
        base_bytes = working_bytes - histogram_cells * 8 + features * 8 + 32
        budget = min(256 * 1024 * 1024, 1024 * 1024 * 1024 - base_bytes)
        bin_capacity = budget // (leaf_count * 8)
        if budget <= 0 or int(compact_spans.max()) > bin_capacity:
            raise ValueError("One feature histogram exceeds the available GPU memory budget.")
        maximum_tile, tile_bins = 0, 0
        for span in compact_spans:
            span = int(span)
            if tile_bins and tile_bins + span > bin_capacity:
                maximum_tile = max(maximum_tile, tile_bins)
                tile_bins = 0
            tile_bins += span
        maximum_tile = max(maximum_tile, tile_bins, 1)
        working_bytes = base_bytes + leaf_count * maximum_tile * 8
    if working_bytes > 1024 * 1024 * 1024:
        raise ValueError("Training exceeds the experimental 1 GiB GPU memory limit.")
    params = SessionParams(TrainParams(rows, features, len(candidate_features), max_bins,
                                      iterations, depth, {"L2": 0, "Cosine": 1, "NewtonL2": 2, "NewtonCosine": 3, "SolarL2": 4, "LOOL2": 5, "SatL2": 6}[score_function],
                                      learning_rate, l2_leaf_reg, bias),
                           objective_ids[objective],
                           leaf_estimation_iterations,
                           {"No": 0, "AnyImprovement": 1, "Armijo": 2}[leaf_estimation_backtracking], 0)
    objective_options = ObjectiveOptions(objective_ids[objective], {"Newton": 0, "Gradient": 1, "Exact": 2, "Simple": 3}[leaf_estimation_method], parameter, 0)
    bootstrap_options = BootstrapOptions(bootstrap_ids[bootstrap_type], int(random_seed) & 0xffffffff,
                                         int(random_seed) >> 32, iteration_offset, temperature, fraction,
                                         regularization, mvs_reg is not None, initial_lambda,
                                         initial_mvs_lambda is not None, 0, 0)
    return params, objective_options, bootstrap_options, (bins, targets, sample_weight, initial_predictions,
                                                         candidate_features, candidate_bins, candidate_types,
                                                         pair_winners, pair_losers, pair_weights)


class Session:
    """Own GPU dataset/workspace buffers across incremental tree construction."""
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, iterations, depth,
                 learning_rate, l2_leaf_reg, bias, score_function, objective="RMSE",
                 sample_weight=None, leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
                 initial_predictions=None, candidate_types=None, objective_param=None,
                 leaf_estimation_method="Newton", bootstrap_type="No", random_seed=0,
                 iteration_offset=0, bagging_temperature=1.0, subsample=1.0,
                 mvs_reg=None, initial_mvs_lambda=None, random_strength=0.0,
                 group_offsets=None, query_beta=1.0, query_lambda=0.01,
                 pair_winners=None, pair_losers=None, pair_weights=None):
        self._handle = ct.c_void_p()
        self._lock = threading.RLock()
        self._completed = 0
        self._permutation_count = 1
        self._permutations_configured = False
        self._feature_penalties_configured = False
        self._lib = None
        self._params, objective_options, bootstrap_options, arrays = _prepare(
            bins, targets, candidate_features, candidate_bins, iterations=iterations, depth=depth,
            learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg, bias=bias,
            score_function=score_function, objective=objective, sample_weight=sample_weight,
            leaf_estimation_iterations=leaf_estimation_iterations,
            leaf_estimation_backtracking=leaf_estimation_backtracking,
            initial_predictions=initial_predictions, candidate_types=candidate_types,
            objective_param=objective_param, leaf_estimation_method=leaf_estimation_method,
            bootstrap_type=bootstrap_type, random_seed=random_seed, iteration_offset=iteration_offset,
            bagging_temperature=bagging_temperature, subsample=subsample, mvs_reg=mvs_reg,
            initial_mvs_lambda=initial_mvs_lambda, random_strength=random_strength,
            group_offsets=group_offsets, query_beta=query_beta, query_lambda=query_lambda,
            pair_winners=pair_winners, pair_losers=pair_losers, pair_weights=pair_weights)
        self.objective = objective
        self._lib = _load(build_library())
        bins, targets, weights, initial, features, borders, types, pair_winners, pair_losers, pair_weights = arrays
        error = ct.create_string_buffer(2048)
        if objective == "PairLogit":
            offsets = None if group_offsets is None else np.ascontiguousarray(group_offsets, dtype=np.uint32)
            pair_options = PairOptions(pair_winners.size, 0 if offsets is None else offsets.size - 1, 0, 0)
            code = self._lib.cbm_session_create_pair(ct.byref(self._params), ct.byref(objective_options),
                ct.byref(pair_options), _u32(pair_winners), _u32(pair_losers), _f32(pair_weights), _u32(offsets),
                _u8(bins), _f32(initial), _u32(features), _u32(borders), _u8(types),
                ct.byref(self._handle), error, len(error))
        elif group_offsets is not None:
            offsets = np.ascontiguousarray(group_offsets, dtype=np.uint32)
            query_options = QueryOptions(offsets.size - 1, query_beta, query_lambda, 0)
            code = self._lib.cbm_session_create_query(ct.byref(self._params), ct.byref(objective_options),
                ct.byref(query_options), _u32(offsets), _u8(bins), _f32(targets), _f32(weights), _f32(initial),
                _u32(features), _u32(borders), _u8(types), ct.byref(self._handle), error, len(error))
        else:
            code = self._lib.cbm_session_create_configured(ct.byref(self._params), ct.byref(objective_options),
                                           _u8(bins), _f32(targets), _f32(weights), _f32(initial),
                                           _u32(features), _u32(borders), _u8(types),
                                           ct.byref(self._handle), error, len(error))
        self._check(code, error)
        try:
            if bootstrap_options.bootstrap_type or iteration_offset or random_strength:
                self._check(self._lib.cbm_session_set_bootstrap(self._handle, ct.byref(bootstrap_options),
                                                              error, len(error)), error)
            if random_strength:
                noise_options = ScoreNoiseOptions(random_strength, 0, 0, 0)
                self._check(self._lib.cbm_session_set_score_noise(self._handle, ct.byref(noise_options),
                                                                error, len(error)), error)
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(code, error):
        if code:
            raise RuntimeError("Metal training failed: " + error.value.decode("utf-8", errors="replace"))

    def _require_open(self):
        if not self._handle.value:
            raise RuntimeError("Training session is closed.")

    @property
    def closed(self):
        return not bool(self._handle.value)

    @property
    def completed_iterations(self):
        return self._completed

    @property
    def bootstrap_state(self):
        with self._lock:
            self._require_open()
            absolute, valid, value = ct.c_uint32(), ct.c_uint32(), ct.c_float()
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_get_bootstrap_state(self._handle, ct.byref(absolute),
                                                                ct.byref(value), ct.byref(valid),
                                                                error, len(error)), error)
            return {"iteration_offset": int(absolute.value),
                    "mvs_lambda": float(value.value) if valid.value else None}

    def configure_feature_penalties(self, ctr_unique_values, model_size_reg=0.5, feature_weights=None, used_features=None):
        with self._lock:
            self._require_open()
            if self._completed or self._feature_penalties_configured:
                raise ValueError("Feature penalties can be configured once before training.")
            features = self._params.train.features
            counts = np.asarray(ctr_unique_values)
            if (counts.shape != (features,) or counts.dtype.kind not in "iu" or (counts < 0).any()
                    or (counts > np.iinfo(np.uint32).max).any()):
                raise ValueError("CTR unique counts must be a uint32-compatible feature vector.")
            counts = np.ascontiguousarray(counts, np.uint32)
            with np.errstate(over="ignore", invalid="ignore"):
                strength = np.float32(model_size_reg)
            if not np.isfinite(strength) or strength < 0:
                raise ValueError("model_size_reg must be finite and nonnegative.")
            weights, used = None, None
            if feature_weights is not None:
                weights = np.asarray(feature_weights)
                if weights.shape != (features,) or weights.dtype.kind not in "biuf":
                    raise ValueError("Feature weights must be a numeric feature vector.")
                with np.errstate(over="ignore", invalid="ignore"):
                    weights = np.ascontiguousarray(weights, np.float32)
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("Feature weights must be finite and nonnegative.")
            if used_features is not None:
                used = np.asarray(used_features)
                if (used.shape != (features,) or used.dtype.kind not in "biu" or ((used != 0) & (used != 1)).any()):
                    raise ValueError("Used-feature flags must be a boolean feature vector.")
                used = np.ascontiguousarray(used, np.uint8)
            options, error = FeaturePenaltyOptions(strength, 0, 0, 0), ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_set_feature_penalties(self._handle, ct.byref(options),
                _u32(counts), _f32(weights), _u8(used), error, len(error)), error)
            self._feature_penalties_configured = True

    @property
    def feature_penalty_state(self):
        with self._lock:
            self._require_open()
            used, error = np.empty(self._params.train.features, np.uint8), ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_copy_feature_penalty_state(self._handle, _u8(used), error, len(error)), error)
            return {"used_features": used}

    def configure_permutations(self, bins_list, initial_predictions=None, mvs_lambdas=None, mvs_valid=None):
        with self._lock:
            self._require_open()
            if self._completed or self._permutations_configured:
                raise ValueError("Permutations can be configured once before training.")
            p = self._params.train
            matrices = np.asarray(bins_list)
            if (matrices.ndim != 3 or not 1 <= matrices.shape[0] <= 64
                    or matrices.shape[1:] != (p.features, p.rows) or matrices.dtype.kind not in "iu"
                    or (matrices < 0).any() or (matrices >= p.bins_per_feature).any()):
                raise ValueError("Permutation bins must have shape (1..64, features, rows) within the original bin range.")
            matrices = np.ascontiguousarray(matrices, np.uint8)
            count = matrices.shape[0]
            cursors = None
            if initial_predictions is not None:
                candidate = np.asarray(initial_predictions)
                if candidate.shape != (count, p.rows) or candidate.dtype.kind not in "biuf":
                    raise ValueError("Permutation raw cursors must have shape (permutations, rows).")
                with np.errstate(over="ignore", invalid="ignore"):
                    cursors = np.ascontiguousarray(candidate, np.float32)
                if not np.isfinite(cursors).all():
                    raise ValueError("Permutation raw cursors must be finite float32 values.")
            if (mvs_lambdas is None) != (mvs_valid is None):
                raise ValueError("MVS permutation values and validity flags must be supplied together.")
            lambdas, valid = None, None
            if mvs_lambdas is not None:
                lambdas, valid = np.asarray(mvs_lambdas), np.asarray(mvs_valid)
                if lambdas.shape != (count,) or lambdas.dtype.kind not in "biuf":
                    raise ValueError("MVS permutation values must be a numeric vector.")
                if (valid.shape != (count,) or valid.dtype.kind not in "biu"
                        or ((valid != 0) & (valid != 1)).any()):
                    raise ValueError("MVS validity flags must be a boolean vector.")
                with np.errstate(over="ignore", invalid="ignore"):
                    lambdas = np.ascontiguousarray(lambdas, np.float32)
                if not np.isfinite(lambdas).all() or (lambdas < 0).any():
                    raise ValueError("MVS values must be finite nonnegative float32 values.")
                valid = np.ascontiguousarray(valid, np.uint8)
            matrix_ptrs = (ct.POINTER(ct.c_uint8) * count)(*(_u8(matrix) for matrix in matrices))
            cursor_ptrs = None if cursors is None else (ct.POINTER(ct.c_float) * count)(*(_f32(cursor) for cursor in cursors))
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_set_permutations(self._handle, count, matrix_ptrs,
                cursor_ptrs, _f32(lambdas), _u8(valid), error, len(error)), error)
            self._permutation_count = count
            self._permutations_configured = True

    def select_permutation(self, index):
        with self._lock:
            self._require_open()
            if isinstance(index, bool) or not isinstance(index, numbers.Integral) or not 0 <= index < self._permutation_count:
                raise ValueError("Search permutation index is out of range.")
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_select_permutation(self._handle, int(index), error, len(error)), error)

    @property
    def permutation_state(self):
        with self._lock:
            self._require_open()
            count = self._permutation_count
            cursors = np.empty((count, self._params.train.rows), np.float32)
            lambdas, valid = np.empty(count, np.float32), np.empty(count, np.uint8)
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_copy_permutation_state(self._handle, count, _f32(cursors),
                _f32(lambdas), _u8(valid), error, len(error)), error)
            return {"predictions": cursors, "mvs_lambdas": lambdas, "mvs_valid": valid}

    @property
    def workspace(self):
        with self._lock:
            self._require_open()
            tiles, hist, peak = ct.c_uint32(), ct.c_uint64(), ct.c_uint64()
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_get_workspace_info(self._handle, ct.byref(tiles), ct.byref(hist),
                ct.byref(peak), error, len(error)), error)
            return {"histogram_tiles": tiles.value, "histogram_bytes": hist.value,
                    "estimated_peak_gpu_bytes": peak.value}

    def predictions(self):
        with self._lock:
            self._require_open()
            result, error = np.empty(self._params.train.rows, np.float32), ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_copy_predictions(self._handle, _f32(result), error, len(error)), error)
            return result

    def append_features(self, bins, candidate_features, candidate_bins, *, candidate_types=None,
                        ctr_unique_values=None, feature_weights=None, feature_flags=None,
                        used_features=None, bins_per_feature=None):
        """Publish new columns in every permutation without restarting the open tree."""
        with self._lock:
            self._require_open()
            p = self._params.train
            matrices = np.asarray(bins)
            if matrices.ndim == 2 and self._permutation_count == 1:
                matrices = matrices[None, :, :]
            if (matrices.ndim != 3 or matrices.shape[0] != self._permutation_count
                    or matrices.shape[2] != p.rows or matrices.shape[1] == 0
                    or matrices.dtype.kind not in "iu" or (matrices < 0).any() or (matrices > 255).any()):
                raise ValueError("Appended bins must have shape (permutations, new features, rows) within [0, 255].")
            new_f = matrices.shape[1]
            features, borders = np.asarray(candidate_features), np.asarray(candidate_bins)
            if (features.ndim != 1 or borders.shape != features.shape
                    or (features.size and (features.dtype.kind not in "iu" or borders.dtype.kind not in "iu"
                        or (features < 0).any() or (features >= new_f).any()
                        or (borders < 0).any() or (borders >= 255).any()))):
                raise ValueError("Appended candidates must be matching vectors with local feature IDs and valid borders.")
            def vector(value, name, default, size, maximum, dtype):
                if value is None:
                    return np.full(size, default, dtype=dtype)
                array = np.asarray(value)
                if (array.shape != (size,) or array.dtype.kind not in "biu"
                        or (array < 0).any() or (array > maximum).any()):
                    raise ValueError(f"{name} must be a matching integer vector in [0, {maximum}].")
                return np.ascontiguousarray(array, dtype=dtype)
            types = vector(candidate_types, "candidate_types", 0, features.size, 1, np.uint8)
            counts = vector(ctr_unique_values, "ctr_unique_values", 0, new_f, np.iinfo(np.uint32).max, np.uint32)
            flags = vector(feature_flags, "feature_flags", 3, new_f, 3, np.uint8)
            used = vector(used_features, "used_features", 0, new_f, 1, np.uint8)
            if ((counts > 0) & (used != 0) & ((flags & 2) == 0)).any():
                raise ValueError("A previously used CTR must be globally registered.")
            if feature_weights is None:
                weights = np.ones(new_f, np.float32)
            else:
                weights = np.asarray(feature_weights)
                if weights.shape != (new_f,) or weights.dtype.kind not in "biuf":
                    raise ValueError("feature_weights must be a numeric vector matching new features.")
                with np.errstate(over="ignore", invalid="ignore"):
                    weights = np.ascontiguousarray(weights, np.float32)
                if not np.isfinite(weights).all() or (weights < 0).any():
                    raise ValueError("feature_weights must be finite nonnegative float32 values.")
            needed_bins = max(p.bins_per_feature, int(matrices.max()) + 1)
            if features.size:
                needed_bins = max(needed_bins, int(np.max(borders.astype(np.uint64) + 2 - types)))
            if bins_per_feature is None:
                bins_per_feature = needed_bins
            if (isinstance(bins_per_feature, bool) or not isinstance(bins_per_feature, numbers.Integral)
                    or not needed_bins <= bins_per_feature <= 256):
                raise ValueError("bins_per_feature must cover all existing/new bins and candidates, at most 256.")
            if ((p.features + new_f) * p.rows > np.iinfo(np.uint32).max
                    or p.candidates + features.size > np.iinfo(np.uint32).max):
                raise ValueError("Appended data exceeds the Metal index limit.")
            matrices = np.ascontiguousarray(matrices, np.uint8)
            features, borders = np.ascontiguousarray(features, np.uint32), np.ascontiguousarray(borders, np.uint32)
            matrix_ptrs = (ct.POINTER(ct.c_uint8) * self._permutation_count)(*(_u8(matrix) for matrix in matrices))
            options = AppendFeatureOptions(self._permutation_count, new_f, features.size, bins_per_feature,
                                           (ct.c_uint32 * 4)(0, 0, 0, 0))
            first, error = ct.c_uint32(), ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_append_features(self._handle, ct.byref(options), matrix_ptrs,
                _u32(features), _u32(borders), _u8(types), _u32(counts), _f32(weights), _u8(flags), _u8(used),
                ct.byref(first), error, len(error)), error)
            p.features += new_f
            p.candidates += features.size
            p.bins_per_feature = bins_per_feature
            return int(first.value)

    def set_feature_activity(self, active_features):
        with self._lock:
            self._require_open()
            active = np.asarray(active_features)
            if (active.shape != (self._params.train.features,) or active.dtype.kind not in "biu"
                    or ((active != 0) & (active != 1)).any()):
                raise ValueError("Feature activity must be a boolean vector covering every current feature.")
            active = np.ascontiguousarray(active, np.uint8)
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_set_feature_activity(self._handle, active.size, _u8(active),
                                                                  error, len(error)), error)

    @property
    def feature_metadata(self):
        with self._lock:
            self._require_open()
            count = self._params.train.features
            counts, weights = np.empty(count, np.uint32), np.empty(count, np.float32)
            flags, used, active = (np.empty(count, np.uint8) for _ in range(3))
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_copy_feature_metadata(self._handle, count, _u32(counts),
                _f32(weights), _u8(flags), _u8(used), _u8(active), error, len(error)), error)
            return {"ctr_unique_values": counts, "feature_weights": weights, "feature_flags": flags,
                    "used_features": used, "active_features": active}

    def begin_tree(self):
        with self._lock:
            self._require_open()
            error = ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_begin_tree(self._handle, error, len(error)), error)

    def grow_tree(self):
        with self._lock:
            self._require_open()
            info, error = StructureInfo(), ct.create_string_buffer(2048)
            self._check(self._lib.cbm_session_grow_tree(self._handle, ct.byref(info), error, len(error)), error)
            return {"depth": int(info.depth), "finished": bool(info.finished), "has_split": bool(info.has_split),
                    "feature": int(info.feature), "bin": int(info.bin), "type": int(info.type),
                    "score": float(info.score), "gain": float(info.gain)}

    def finish_tree(self):
        return self._complete_tree("cbm_session_finish_tree")

    def step(self):
        return self._complete_tree("cbm_session_step")

    def _complete_tree(self, operation):
        with self._lock:
            self._require_open()
            p = self._params.train
            if self._completed >= p.iterations:
                raise RuntimeError("Training session has no remaining iterations.")
            features, bins = np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint32)
            types = np.zeros(p.depth, np.uint8)
            values, weights = np.zeros(1 << p.depth, np.float32), np.zeros(1 << p.depth, np.float32)
            info, depth, error = StepInfo(), ct.c_uint32(), ct.create_string_buffer(2048)
            code = getattr(self._lib, operation)(self._handle, ct.byref(info), ct.byref(depth),
                                              _u32(features), _u32(bins), _u8(types),
                                              _f32(values), _f32(weights), error, len(error))
            self._check(code, error)
            self._completed = int(info.completed_iterations)
            count = int(depth.value)
            statistics = _stats(info.stats)
            statistics.update(self.workspace)
            return StepResult(self._completed, bool(info.finished), count,
                              features[:count].copy(), bins[:count].copy(), types[:count].copy(),
                              values[:1 << count].copy(), weights[:1 << count].copy(),
                              float(info.loss), statistics)

    def result(self):
        with self._lock:
            self._require_open()
            p, count = self._params.train, self._completed
            if count * (1 << p.depth) * 8 > 512 * 1024 * 1024:
                raise ValueError("Padded result exceeds 512 MiB; consume incremental step outputs.")
            result = TrainResult(np.zeros(count, np.uint32), np.zeros((count, p.depth), np.uint32),
                                 np.zeros((count, p.depth), np.uint32),
                                 np.zeros((count, 1 << p.depth), np.float32),
                                 np.zeros((count, 1 << p.depth), np.float32),
                                 np.zeros(p.rows, np.float32), np.zeros(count + 1, np.float32), {},
                                 np.zeros((count, p.depth), np.uint8))
            completed, stats, error = ct.c_uint32(), TrainStats(), ct.create_string_buffer(2048)
            code = self._lib.cbm_session_result(
                self._handle, count, ct.byref(completed), _u32(result.depths),
                _u32(result.split_features), _u32(result.split_bins), _u8(result.split_types),
                _f32(result.leaf_values), _f32(result.leaf_weights), _f32(result.predictions),
                _f32(result.rmse), ct.byref(stats), error, len(error))
            self._check(code, error)
            if completed.value != count:
                raise RuntimeError("Native session returned an inconsistent completed tree count.")
            result.stats = _stats(stats)
            result.stats["bootstrap_state"] = self.bootstrap_state
            result.stats.update(self.workspace)
            return result

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._handle.value and self._lib is not None:
                self._lib.cbm_session_close(self._handle)
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


def train(bins, targets, candidate_features, candidate_bins, *, iterations, depth,
          learning_rate, l2_leaf_reg, bias, score_function, objective="RMSE", sample_weight=None,
          leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
          initial_predictions=None, candidate_types=None, objective_param=None,
          leaf_estimation_method="Newton", bootstrap_type="No", random_seed=0,
          iteration_offset=0, bagging_temperature=1.0, subsample=1.0,
          mvs_reg=None, initial_mvs_lambda=None, random_strength=0.0,
          group_offsets=None, query_beta=1.0, query_lambda=0.01,
          pair_winners=None, pair_losers=None, pair_weights=None):
    """Compatibility entry point, implemented using the persistent GPU session."""
    with Session(bins, targets, candidate_features, candidate_bins, iterations=iterations, depth=depth,
                 learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg, bias=bias,
                 score_function=score_function, objective=objective, sample_weight=sample_weight,
                 leaf_estimation_iterations=leaf_estimation_iterations,
                 leaf_estimation_backtracking=leaf_estimation_backtracking,
                 initial_predictions=initial_predictions, candidate_types=candidate_types,
                 objective_param=objective_param, leaf_estimation_method=leaf_estimation_method,
                 bootstrap_type=bootstrap_type, random_seed=random_seed, iteration_offset=iteration_offset,
                 bagging_temperature=bagging_temperature, subsample=subsample, mvs_reg=mvs_reg,
                 initial_mvs_lambda=initial_mvs_lambda, random_strength=random_strength,
                 group_offsets=group_offsets, query_beta=query_beta, query_lambda=query_lambda,
                 pair_winners=pair_winners, pair_losers=pair_losers, pair_weights=pair_weights) as session:
        for _ in range(iterations):
            session.step()
        return session.result()
