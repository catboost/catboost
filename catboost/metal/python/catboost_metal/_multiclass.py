"""Coupled CUDA-derived multiclass training on Metal; no CPU fitting path."""
import ctypes as ct
import copy
import functools
import hashlib
import numbers
from pathlib import Path
import platform
import subprocess
import threading

import numpy as np

from . import _greedy

from ._native import (TrainResult, StepResult, StepInfo, TrainStats, BootstrapOptions, ScoreNoiseOptions, FeaturePenaltyOptions,
                      _u8, _u32, _f32, _stats)


class Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "candidates", "bins_per_feature", "classes", "objective",
        "iterations", "depth", "score_function", "leaf_method", "leaf_iterations", "reserved")]
    _fields_ += [("learning_rate", ct.c_float), ("l2", ct.c_float),
                 ("reserved1", ct.c_uint32), ("reserved2", ct.c_uint32)]


class GreedyOptions(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("policy", "max_leaves", "min_data_in_leaf", "reserved")]


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Multiclass Metal training requires macOS on Apple Silicon.")
    import fcntl
    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    names = ("metal_multiclass.mm", "metal_multiclass.h", "metal_multiclass_math.h",
             "metal_multiclass_kernels.h", "metal_kernels.h", "metal_trainer.h",
             "metal_additional_objective_kernels.h", "metal_objective_kernels.h",
             "metal_histogram_kernels.h", "metal_incremental_partition_kernels.h",
             "metal_deep_partition_kernels.h", "metal_bootstrap_kernels.h", "metal_score_noise_kernels.h",
             "metal_multiclass_bootstrap.h", "metal_multiclass_backtracking.h", "metal_kernel_abi.h",
             "metal_multioutput_math_kernels.h", "metal_multiclass_scores.h", "metal_greedy_trainer.h",
             "metal_greedy_kernels.h", "metal_greedy_bootstrap_kernels.h", "metal_greedy_vector_scores.h")
    digest = hashlib.sha256(platform.platform().encode())
    for name in names:
        digest.update(name.encode()); digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_multiclass_{digest.hexdigest()[:20]}.dylib"
    with (build / "multiclass-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                completed = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc",
                    "-dynamiclib", "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_multiclass.mm"), "-o", str(temporary)], capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError("Could not build multiclass Metal runtime:\n" + completed.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    lib = ct.CDLL(str(path))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    text = [ct.c_char_p, ct.c_size_t]
    lib.cbm_multiclass_session_create.argtypes = [ct.POINTER(Params), u8, u32, f32, f32,
                                                 u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_multioutput_session_create.argtypes = [ct.POINTER(Params), u8, f32, f32, f32,
                                                 u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_multioutput_session_create.restype = ct.c_int
    lib.cbm_multiclass_session_create_greedy.argtypes = [ct.POINTER(Params), ct.POINTER(GreedyOptions),
        u8, u32, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_multiclass_session_create_greedy.restype = ct.c_int
    lib.cbm_multiclass_session_step_greedy.argtypes = [ct.c_void_p, ct.POINTER(_greedy.StepInfo),
        ct.POINTER(_greedy.Node), f32, f32] + text
    lib.cbm_multiclass_session_step_greedy.restype = ct.c_int
    lib.cbm_multiclass_session_step.argtypes = [ct.c_void_p, ct.POINTER(StepInfo), u32, u32,
                                               u32, u8, f32, f32] + text
    lib.cbm_multiclass_session_copy_predictions.argtypes = [ct.c_void_p, f32] + text
    lib.cbm_multiclass_session_info.argtypes = [ct.c_void_p, ct.POINTER(StepInfo)] + text
    lib.cbm_multiclass_session_set_bootstrap.argtypes = [ct.c_void_p, ct.POINTER(BootstrapOptions)] + text
    lib.cbm_multiclass_session_get_bootstrap_state.argtypes = [ct.c_void_p, u32, f32, u32] + text
    lib.cbm_multiclass_session_set_score_noise.argtypes = [ct.c_void_p, ct.POINTER(ScoreNoiseOptions)] + text
    lib.cbm_multiclass_session_set_backtracking.argtypes = [ct.c_void_p, ct.c_uint32] + text
    lib.cbm_multiclass_session_set_permutations.argtypes = [ct.c_void_p, ct.c_uint32, ct.POINTER(u8),
                                                          ct.POINTER(f32), f32, u8] + text
    lib.cbm_multiclass_session_select_permutation.argtypes = [ct.c_void_p, ct.c_uint32] + text
    lib.cbm_multiclass_session_copy_permutation_state.argtypes = [ct.c_void_p, ct.c_uint32, f32, f32, u8] + text
    lib.cbm_multiclass_session_copy_optimization_state.argtypes = [ct.c_void_p, ct.c_uint32, f32] + text
    lib.cbm_multiclass_session_restore_optimization_state.argtypes = [ct.c_void_p, ct.c_uint32, f32] + text
    lib.cbm_multiclass_session_set_feature_penalties.argtypes = [ct.c_void_p, ct.POINTER(FeaturePenaltyOptions), u32, f32, u8] + text
    lib.cbm_multiclass_session_copy_feature_penalty_state.argtypes = [ct.c_void_p, u8] + text
    lib.cbm_multiclass_session_close.argtypes = [ct.c_void_p]
    lib.cbm_multiclass_session_close.restype = None
    lib.cbm_multiclass_math.argtypes = [ct.c_uint32] * 5 + [ct.c_float, u32, f32, f32, u32,
                                                         f32, f32, f32, f32, ct.POINTER(TrainStats)] + text
    for name in ("create", "step", "copy_predictions", "info", "set_bootstrap", "get_bootstrap_state",
                 "set_score_noise", "set_backtracking", "set_permutations", "select_permutation", "copy_permutation_state",
                 "copy_optimization_state", "restore_optimization_state", "set_feature_penalties", "copy_feature_penalty_state"):
        getattr(lib, "cbm_multiclass_session_" + name).restype = ct.c_int
    lib.cbm_multiclass_math.restype = ct.c_int
    return lib


def _integer(name, value, low, high):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}].")
    return int(value)


def _finite_array(name, value, shape):
    value = np.asarray(value)
    if value.shape != shape or value.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a numeric array with shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.array(value, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite float32 values.")
    return value


def _labels(targets, rows, classes):
    targets = np.asarray(targets)
    if (targets.shape != (rows,) or targets.dtype.kind not in "biuf" or not np.isfinite(targets).all()
            or (targets < 0).any() or (targets >= classes).any() or (targets != np.floor(targets)).any()):
        raise ValueError("targets must be integer class indices in [0, classes).")
    return np.ascontiguousarray(targets, dtype=np.uint32)


def _weights(value, rows):
    result = np.ones(rows, np.float32) if value is None else _finite_array("sample_weight", value, (rows,))
    if (result < 0).any() or not 0 < result.sum(dtype=np.float64) < 1e30:
        raise ValueError("sample_weight must be nonnegative with positive total below 1e30.")
    return result


def _objective(value):
    names = ("MultiClass", "MultiClassOneVsAll", "MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")
    if value not in names:
        raise ValueError("Unsupported vector objective.")
    return names.index(value)


def _float_targets(targets, rows, dimensions, objective_id):
    shape = (rows,) if objective_id == 3 else (rows, dimensions)
    targets = _finite_array("targets", targets, shape)
    if objective_id == 3 and dimensions != 2:
        raise ValueError("RMSEWithUncertainty requires two output dimensions.")
    if objective_id >= 4 and ((targets < 0).any() or (targets > 1).any()):
        raise ValueError("Multilabel targets must be in [0,1].")
    if objective_id == 4 and ((targets != 0) & (targets != 1)).any():
        raise ValueError("MultiLogloss targets must be binary.")
    return np.ascontiguousarray(targets.T, np.float32)


def _method(value):
    if value not in ("Newton", "Gradient"):
        raise ValueError("leaf_estimation_method must be Newton or Gradient.")
    return int(value == "Gradient")


def _check(code, error):
    if code:
        raise RuntimeError("Metal multiclass training failed: " + error.value.decode("utf-8", errors="replace"))


def objective_and_leaf_directions(logits, targets, *, classes, leaf_ids=None, leaves=None,
                                  objective="MultiClass", sample_weight=None, l2_leaf_reg=3.,
                                  leaf_estimation_method="Newton"):
    """Actual GPU math diagnostic. Logits/gradients are class-major[D,N].

    Returns gradients[D,N], probabilities[C,N], weighted_losses[N], and
    directions[leaves,D]. D=C-1 for MultiClass and C for OneVsAll.
    """
    classes = _integer("classes", classes, 2, 64)
    objective_id, method = _objective(objective), _method(leaf_estimation_method)
    if objective_id >= 2:
        raise ValueError("Use _multioutput_math for float-target objective diagnostics.")
    dimensions = classes - int(objective_id == 0)
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[0] != dimensions:
        raise ValueError("logits must be class-major[D,rows].")
    rows = _integer("rows", logits.shape[1], 1, 16777216)
    logits = _finite_array("logits", logits, (dimensions, rows))
    targets, weights = _labels(targets, rows, classes), _weights(sample_weight, rows)
    if leaf_ids is None:
        leaf_ids = np.zeros(rows, np.uint32)
    else:
        leaf_ids = np.asarray(leaf_ids)
        if leaf_ids.shape != (rows,) or leaf_ids.dtype.kind not in "iu" or (leaf_ids < 0).any():
            raise ValueError("leaf_ids must be a nonnegative integer vector.")
    leaves = _integer("leaves", int(leaf_ids.max()) + 1 if leaves is None else leaves, 1, 65536)
    if (leaf_ids >= leaves).any():
        raise ValueError("leaf_ids exceed the leaf count.")
    leaf_ids = np.ascontiguousarray(leaf_ids, np.uint32)
    l2 = _finite_array("l2_leaf_reg", l2_leaf_reg, ())
    if l2 < 0:
        raise ValueError("l2_leaf_reg must be nonnegative.")
    gradients = np.empty_like(logits); probabilities = np.empty((classes, rows), np.float32)
    losses = np.empty(rows, np.float32); directions = np.empty((leaves, dimensions), np.float32)
    lib, stats, error = _load(build_library()), TrainStats(), ct.create_string_buffer(2048)
    _check(lib.cbm_multiclass_math(rows, classes, objective_id, leaves, method, float(l2),
           _u32(targets), _f32(weights), _f32(logits), _u32(leaf_ids), _f32(gradients),
           _f32(probabilities), _f32(losses), _f32(directions), ct.byref(stats), error, len(error)), error)
    return dict(gradients=gradients, probabilities=probabilities, weighted_losses=losses,
                directions=directions, stats=_stats(stats))


class Session:
    """Persistent quantized dataset, class cursors, histograms and leaf workspace."""
    def __init__(self, bins, targets, candidate_features, candidate_bins, *, classes,
                 iterations=100, depth=6, learning_rate=.03, l2_leaf_reg=3., bias=0.,
                 score_function="Cosine", objective="MultiClass", sample_weight=None,
                 leaf_estimation_iterations=1, leaf_estimation_backtracking="No",
                 initial_predictions=None, candidate_types=None, leaf_estimation_method="Newton",
                 bootstrap_type="No", random_strength=0., random_seed=0, iteration_offset=0,
                 objective_param=None, bagging_temperature=1., subsample=1., mvs_reg=None,
                 initial_mvs_lambda=None, initial_optimization_predictions=None,
                 grow_policy="SymmetricTree", max_leaves=31, min_data_in_leaf=1):
        self._handle = ct.c_void_p(); self._lib = None; self._lock = threading.RLock()
        self._steps = []
        self._permutation_count = 1
        self._permutations_configured = False
        classes = _integer("classes", classes, 2, 64)
        iterations = _integer("iterations", iterations, 1, 100000)
        policies = {"Depthwise": 0, "Lossguide": 1, "Region": 2}
        if grow_policy != "SymmetricTree" and grow_policy not in policies:
            raise ValueError("grow_policy must be SymmetricTree, Depthwise, Lossguide or Region.")
        self._greedy_options = None
        if grow_policy != "SymmetricTree":
            depth = _integer("depth", depth, 0, 16 if grow_policy == "Depthwise" else 65535 if grow_policy == "Region" else 2**32-1)
            max_leaves = _integer("max_leaves", max_leaves, 1, 65536)
            min_data_in_leaf = _integer("min_data_in_leaf", min_data_in_leaf, 0, 2**32-1)
            # CUDA Depthwise and Region derive capacity from depth; only Lossguide uses max_leaves.
            capacity = 1 << depth if grow_policy == "Depthwise" else depth + 1 if grow_policy == "Region" else min(max_leaves, 1 << min(depth, 16))
            self._greedy_options = GreedyOptions(policies[grow_policy], capacity, min_data_in_leaf, 0)
        else:
            depth = _integer("depth", depth, 0, 16)
        leaf_iterations = _integer("leaf_estimation_iterations", leaf_estimation_iterations, 1, 1000)
        self._iteration_offset = _integer("iteration_offset", iteration_offset, 0, 2**32 - 1 - iterations)
        _integer("random_seed", random_seed, 0, 2**64 - 1)
        objective_id, method = _objective(objective), _method(leaf_estimation_method)
        if self._greedy_options is not None and objective_id not in (0, 1, 3):
            raise ValueError("CUDA greedy vector objectives are MultiClass, MultiClassOneVsAll and RMSEWithUncertainty.")
        score_ids = {"L2": 0, "Cosine": 1, "SolarL2": 4, "LOOL2": 5, "SatL2": 6}
        if score_function not in score_ids:
            raise ValueError("Vector score_function must be L2, Cosine, SolarL2, LOOL2 or SatL2.")
        if leaf_estimation_backtracking not in ("No", "AnyImprovement", "Armijo"):
            raise ValueError("leaf_estimation_backtracking must be No, AnyImprovement or Armijo.")
        if bootstrap_type not in ("No", "Bayesian", "Bernoulli", "Poisson"):
            raise ValueError("Multiclass supports No/Bayesian/Bernoulli/Poisson bootstrap; CUDA rejects multiclass MVS.")
        if objective_param is not None or initial_mvs_lambda is not None:
            raise ValueError("Multiclass does not accept an objective parameter or MVS state.")
        sampling = _finite_array("sampling parameters", [bagging_temperature, subsample, random_strength], (3,))
        if sampling[0] < 0 or not 0 < sampling[1] <= 1 or sampling[2] < 0 or (bootstrap_type == "Poisson" and sampling[1] == 1):
            raise ValueError("Invalid sampling parameters or random_strength.")
        if mvs_reg is not None:
            regularization = _finite_array("mvs_reg", mvs_reg, ())
            if regularization < 0:
                raise ValueError("mvs_reg must be nonnegative.")
        bins = np.asarray(bins)
        if bins.ndim != 2 or bins.dtype.kind not in "iu" or not all(bins.shape):
            raise ValueError("bins must be a nonempty two-dimensional integer array.")
        features, rows = bins.shape
        _integer("rows", rows, 1, 16777216)
        if (bins < 0).any() or (bins > 255).any() or rows * features > 2**32 - 1:
            raise ValueError("Quantized bins must be in [0,255], with at most uint32 cells.")
        bins = np.ascontiguousarray(bins, np.uint8)
        targets = (_labels(targets, rows, classes) if objective_id < 2
                   else _float_targets(targets, rows, classes, objective_id))
        weights = _weights(sample_weight, rows)
        cf, cb = np.asarray(candidate_features), np.asarray(candidate_bins)
        if cf.ndim != 1 or cf.shape != cb.shape:
            raise ValueError("Candidate arrays must be matching vectors.")
        if cf.size and (cf.dtype.kind not in "iu" or cb.dtype.kind not in "iu" or
                       (cf < 0).any() or (cf >= features).any() or (cb < 0).any() or (cb > 255).any()):
            raise ValueError("Invalid candidate feature or bin index.")
        if candidate_types is None:
            candidate_types = np.zeros(cf.size, np.uint8)
        candidate_types = np.asarray(candidate_types)
        if (candidate_types.shape != cf.shape or candidate_types.dtype.kind not in "biu"
                or ((candidate_types != 0) & (candidate_types != 1)).any()):
            raise ValueError("candidate_types must contain numeric(0)/one-hot(1) flags.")
        if ((candidate_types == 0) & (cb == 255)).any():
            raise ValueError("A numeric border must be below 255.")
        type_by_feature = {}
        for feature, flag in zip(cf, candidate_types):
            if type_by_feature.setdefault(int(feature), int(flag)) != flag:
                raise ValueError("A feature cannot mix numeric and one-hot candidates.")
        cf, cb = np.ascontiguousarray(cf, np.uint32), np.ascontiguousarray(cb, np.uint32)
        candidate_types = np.ascontiguousarray(candidate_types, np.uint8)
        bins_per_feature = int(bins.max()) + 1
        if cb.size:
            bins_per_feature = max(bins_per_feature, int((cb + np.where(candidate_types, 1, 2)).max()))
        scalars = _finite_array("training scalars", [learning_rate, l2_leaf_reg], (2,))
        if not 0 < scalars[0] <= 1 or scalars[1] < 0:
            raise ValueError("learning_rate must be in (0,1] and l2_leaf_reg nonnegative.")
        bias_array = np.asarray(bias)
        if bias_array.shape == ():
            bias_array = np.full(classes, bias)
        bias_array = _finite_array("bias", bias_array, (classes,))
        initial = (_finite_array("initial_predictions", initial_predictions, (rows, classes))
                   if initial_predictions is not None else np.tile(bias_array, (rows, 1)))
        active = None if initial_optimization_predictions is None else _finite_array(
            "initial_optimization_predictions", initial_optimization_predictions, (classes - int(objective_id == 0), rows))
        if active is not None and initial_predictions is None:
            raise ValueError("Exact optimizer restoration also requires published initial_predictions.")
        self._params = Params(rows, features, len(cf), bins_per_feature, classes, objective_id,
                              iterations, depth, score_ids[score_function], method,
                              leaf_iterations, 0, *map(float, scalars), 0, 0)
        self.objective = objective
        self._lib = _load(build_library())
        error = ct.create_string_buffer(2048)
        if self._greedy_options is not None:
            _check(self._lib.cbm_multiclass_session_create_greedy(ct.byref(self._params), ct.byref(self._greedy_options),
                _u8(bins), _u32(targets) if objective_id < 2 else None, _f32(targets) if objective_id >= 2 else None,
                _f32(weights), _f32(initial), _u32(cf), _u32(cb), _u8(candidate_types),
                ct.byref(self._handle), error, len(error)), error)
        else:
            create = self._lib.cbm_multiclass_session_create if objective_id < 2 else self._lib.cbm_multioutput_session_create
            _check(create(ct.byref(self._params), _u8(bins), _u32(targets) if objective_id < 2 else _f32(targets),
                   _f32(weights), _f32(initial), _u32(cf), _u32(cb), _u8(candidate_types),
                   ct.byref(self._handle), error, len(error)), error)
        try:
            bootstrap = BootstrapOptions(("No", "Bayesian", "Bernoulli", "Poisson").index(bootstrap_type),
                random_seed & 0xffffffff, random_seed >> 32, iteration_offset,
                float(sampling[0]), float(sampling[1]), 0, 0, 0, 0, 0, 0)
            noise = ScoreNoiseOptions(float(sampling[2]), 0, 0, 0)
            _check(self._lib.cbm_multiclass_session_set_bootstrap(self._handle, ct.byref(bootstrap), error, len(error)), error)
            _check(self._lib.cbm_multiclass_session_set_score_noise(self._handle, ct.byref(noise), error, len(error)), error)
            _check(self._lib.cbm_multiclass_session_set_backtracking(self._handle,
                ("No", "AnyImprovement", "Armijo").index(leaf_estimation_backtracking), error, len(error)), error)
            if active is not None:
                _check(self._lib.cbm_multiclass_session_restore_optimization_state(self._handle, 1, _f32(active), error, len(error)), error)
        except Exception:
            self.close()
            raise
        self._initial_loss = self._info().loss

    def _require_open(self):
        if not self._handle.value:
            raise RuntimeError("Multiclass training session is closed.")

    def _info(self):
        self._require_open()
        info, error = StepInfo(), ct.create_string_buffer(2048)
        _check(self._lib.cbm_multiclass_session_info(self._handle, ct.byref(info), error, len(error)), error)
        return info

    @property
    def closed(self):
        return not bool(self._handle.value)

    @property
    def completed_iterations(self):
        return len(self._steps)

    @property
    def bootstrap_state(self):
        with self._lock:
            self._require_open()
            iteration, value, valid, error = ct.c_uint32(), ct.c_float(), ct.c_uint32(), ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_get_bootstrap_state(self._handle, ct.byref(iteration),
                   ct.byref(value), ct.byref(valid), error, len(error)), error)
            return {"iteration_offset": int(iteration.value), "mvs_lambda": float(value.value) if valid.value else None}

    def predictions(self):
        with self._lock:
            self._require_open()
            result = np.empty((self._params.rows, self._params.classes), np.float32)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_copy_predictions(self._handle, _f32(result), error, len(error)), error)
            return result

    def configure_permutations(self, bins_list, initial_predictions=None, mvs_lambdas=None, mvs_valid=None,
                               optimization_predictions=None):
        """Keep separate class cursors while sharing each selected tree structure."""
        with self._lock:
            self._require_open()
            if self._steps or self._permutations_configured:
                raise ValueError("Permutations can be configured once before training.")
            p = self._params
            matrices = np.asarray(bins_list)
            if (matrices.ndim != 3 or matrices.shape[1:] != (p.features, p.rows)
                    or matrices.dtype.kind not in "iu" or not 1 <= matrices.shape[0] <= 64
                    or (matrices < 0).any() or (matrices >= p.bins_per_feature).any()):
                raise ValueError("permutation bins must be integer[P,features,rows] on the shared feature grid.")
            count = matrices.shape[0]
            matrices = np.ascontiguousarray(matrices, np.uint8)
            initial = None if initial_predictions is None else _finite_array(
                "permutation initial_predictions", initial_predictions, (count, p.rows, p.classes))
            active = None if optimization_predictions is None else _finite_array("optimization_predictions",
                optimization_predictions, (count, p.classes - int(p.objective == 0), p.rows))
            if active is not None and initial is None:
                raise ValueError("Exact optimizer restoration also requires published initial_predictions.")
            if initial is not None and p.objective == 0:
                with np.errstate(over="ignore", invalid="ignore"):
                    differences = initial[:, :, :-1] - initial[:, :, -1:]
                if not np.isfinite(differences).all():
                    raise ValueError("Permutation gauge differences must fit float32.")
            if (mvs_lambdas is None) != (mvs_valid is None):
                raise ValueError("MVS placeholder arrays must be supplied together.")
            lambdas = np.zeros(count, np.float32)
            valid = np.zeros(count, np.uint8)
            if mvs_lambdas is not None:
                lambdas = _finite_array("mvs_lambdas", mvs_lambdas, (count,))
                flags = np.asarray(mvs_valid)
                if flags.shape != (count,) or flags.dtype.kind not in "biu" or flags.any() or lambdas.any():
                    raise ValueError("Multiclass MVS placeholder values and validity flags must be zero.")
            u8, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_float)
            bin_pointers = (u8 * count)(*[_u8(matrix) for matrix in matrices])
            prediction_pointers = None if initial is None else (f32 * count)(*[_f32(matrix) for matrix in initial])
            error = ct.create_string_buffer(2048)
            code = self._lib.cbm_multiclass_session_set_permutations(self._handle, count, bin_pointers,
                prediction_pointers, _f32(lambdas), _u8(valid), error, len(error))
            if code:
                self.close()
            _check(code, error)
            if active is not None:
                code = self._lib.cbm_multiclass_session_restore_optimization_state(self._handle, count, _f32(active), error, len(error))
                if code:
                    self.close()
                _check(code, error)
            self._permutation_count = count
            self._permutations_configured = True
            self._initial_loss = self._info().loss

    def select_permutation(self, index):
        with self._lock:
            self._require_open()
            if not self._permutations_configured:
                raise ValueError("Configure multiclass permutations before selecting one.")
            index = _integer("permutation index", index, 0, self._permutation_count - 1)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_select_permutation(self._handle, index, error, len(error)), error)

    @property
    def permutation_state(self):
        with self._lock:
            self._require_open()
            p, count = self._params, self._permutation_count
            predictions = np.empty((count, p.rows, p.classes), np.float32)
            lambdas, valid = np.empty(count, np.float32), np.empty(count, np.uint8)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_copy_permutation_state(self._handle, count,
                _f32(predictions), _f32(lambdas), _u8(valid), error, len(error)), error)
            return dict(predictions=predictions, mvs_lambdas=lambdas, mvs_valid=valid,
                        optimization_predictions=self.optimization_predictions(all_permutations=True))

    def optimization_predictions(self, *, all_permutations=False):
        """Exact class-major optimizer cursor; optionally return every permutation."""
        with self._lock:
            self._require_open()
            p, count = self._params, self._permutation_count
            result = np.empty((count, p.classes - int(p.objective == 0), p.rows), np.float32)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_copy_optimization_state(self._handle, count, _f32(result), error, len(error)), error)
            return result if all_permutations else result[-1]

    def configure_feature_penalties(self, ctr_unique_values, model_size_reg=.5, feature_weights=None, used_features=None):
        """CUDA greedy CTR size penalty and forest-wide used-feature state."""
        with self._lock:
            self._require_open()
            if self.completed_iterations:
                raise ValueError("Feature penalties must be configured before training.")
            count = self._params.features
            counts = np.asarray(ctr_unique_values)
            if (counts.shape != (count,) or counts.dtype.kind not in "iu" or
                    (counts < 0).any() or (counts > np.iinfo(np.uint32).max).any()):
                raise ValueError("ctr_unique_values must be a nonnegative uint32 feature vector.")
            counts = np.ascontiguousarray(counts, np.uint32)
            regularization = _finite_array("model_size_reg", model_size_reg, ())
            if regularization < 0:
                raise ValueError("model_size_reg must be nonnegative.")
            weights = None if feature_weights is None else _finite_array("feature_weights", feature_weights, (count,))
            if weights is not None and (weights < 0).any():
                raise ValueError("feature_weights must be nonnegative.")
            flags = None
            if used_features is not None:
                flags = np.asarray(used_features)
                if flags.shape != (count,) or flags.dtype.kind not in "biu" or ((flags != 0) & (flags != 1)).any():
                    raise ValueError("used_features must contain one zero/one flag per feature.")
                flags = np.ascontiguousarray(flags, np.uint8)
            options, error = FeaturePenaltyOptions(float(regularization), 0, 0, 0), ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_set_feature_penalties(self._handle, ct.byref(options),
                _u32(counts), _f32(weights), _u8(flags), error, len(error)), error)

    @property
    def feature_penalty_state(self):
        with self._lock:
            self._require_open()
            used, error = np.empty(self._params.features, np.uint8), ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_copy_feature_penalty_state(self._handle, _u8(used), error, len(error)), error)
            return {"used_features": used}

    def step(self):
        with self._lock:
            self._require_open()
            p = self._params
            if self.completed_iterations >= p.iterations:
                raise RuntimeError("Multiclass session has no remaining iterations.")
            if self._greedy_options is not None:
                capacity = self._greedy_options.max_leaves
                nodes = np.zeros((2 * capacity - 1, 6), np.uint32)
                values, weights = np.zeros((capacity, p.classes), np.float32), np.zeros(capacity, np.float32)
                info, error = _greedy.StepInfo(), ct.create_string_buffer(2048)
                _check(self._lib.cbm_multiclass_session_step_greedy(self._handle, ct.byref(info),
                    nodes.ctypes.data_as(ct.POINTER(_greedy.Node)), _f32(values), _f32(weights), error, len(error)), error)
                if not (1 <= info.node_count <= len(nodes) and 1 <= info.leaf_count <= capacity):
                    raise RuntimeError("Metal returned invalid vector greedy output sizes.")
                result = _greedy.StepResult(int(info.completed_iterations), bool(info.finished),
                    nodes[:info.node_count].copy(), values[:info.leaf_count].copy(), weights[:info.leaf_count].copy(),
                    float(info.loss), _stats(info.stats))
                self._steps.append(copy.deepcopy(result))
                return result
            features, borders, types = np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint32), np.zeros(p.depth, np.uint8)
            values = np.zeros((1 << p.depth, p.classes), np.float32)
            weights = np.zeros(1 << p.depth, np.float32)
            info, depth, error = StepInfo(), ct.c_uint32(), ct.create_string_buffer(2048)
            _check(self._lib.cbm_multiclass_session_step(self._handle, ct.byref(info), ct.byref(depth),
                   _u32(features), _u32(borders), _u8(types), _f32(values), _f32(weights), error, len(error)), error)
            count = int(depth.value)
            # Match scalar StepResult: compact actual-depth arrays. result()
            # expands these to the requested fixed tree stride.
            result = StepResult(int(info.completed_iterations), bool(info.finished), count,
                features[:count].copy(), borders[:count].copy(), types[:count].copy(),
                values[:1 << count].copy(), weights[:1 << count].copy(), float(info.loss), _stats(info.stats))
            self._steps.append(copy.deepcopy(result))
            return result

    def result(self):
        with self._lock:
            self._require_open()
            p, count = self._params, len(self._steps)
            if self._greedy_options is not None:
                result = _greedy.TrainResult(tuple(copy.deepcopy(self._steps)), self.predictions(),
                    np.array([self._initial_loss, *[step.loss for step in self._steps]], np.float32), _stats(self._info().stats))
                result.stats["bootstrap_state"] = self.bootstrap_state
                return result
            if count * (1 << p.depth) * (p.classes + 1) * 4 > 512 * 1024**2:
                raise ValueError("Padded multiclass output exceeds 512 MiB; consume incremental steps.")
            result = TrainResult(np.zeros(count, np.uint32), np.zeros((count, p.depth), np.uint32),
                np.zeros((count, p.depth), np.uint32), np.zeros((count, 1 << p.depth, p.classes), np.float32),
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
            result.stats["bootstrap_state"] = self.bootstrap_state
            return result

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if self._handle.value and self._lib is not None:
                    self._lib.cbm_multiclass_session_close(self._handle)
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
    counts = kwargs.pop("ctr_unique_values", None)
    regularization = kwargs.pop("model_size_reg", .5)
    weights = kwargs.pop("feature_weights", None)
    used = kwargs.pop("used_features", None)
    with Session(bins, targets, candidate_features, candidate_bins, **kwargs) as session:
        if counts is not None:
            session.configure_feature_penalties(counts, regularization, weights, used)
        elif weights is not None or used is not None:
            session.configure_feature_penalties(np.zeros(session._params.features, np.uint32), regularization, weights, used)
        for _ in range(session._params.iterations):
            session.step()
        return session.result()
