"""Private quantized Metal sessions for scalar non-symmetric tree training.

This greedy session supports scalar objectives with Plain boosting and GPU
sampling, scoring, and leaf estimation. It does not call a CatBoost CPU trainer.
"""
import ctypes as ct
from dataclasses import dataclass
import functools
import hashlib
import numbers
from pathlib import Path
import platform
import re
import subprocess
import threading

import numpy as np

from ._native import BootstrapOptions, ObjectiveOptions, QueryOptions, PairOptions, ScoreNoiseOptions, TrainStats, _u8, _u32, _f32, _stats


MISSING_LEAF = np.iinfo(np.uint32).max
OBJECTIVES = ("RMSE", "Logloss", "CrossEntropy", "Poisson", "Huber", "Expectile",
              "Lq", "Tweedie", "LogLinQuantile", "Quantile", "MAE", "MAPE", "QueryRMSE", "QuerySoftMax", "PairLogit")
SCORES = ("L2", "Cosine", "NewtonL2", "NewtonCosine", "SolarL2", "LOOL2", "SatL2")
BOOTSTRAPS = ("No", "Bayesian", "Bernoulli", "Poisson")
OBJECTIVE_PARAMETERS = {"Huber": "delta", "Expectile": "alpha", "Lq": "q",
                        "Tweedie": "variance_power", "LogLinQuantile": "alpha", "Quantile": "alpha"}


class Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "rows", "features", "candidates", "bins_per_feature", "iterations", "depth",
        "max_leaves", "min_data_in_leaf", "policy", "objective", "score_function",
        "leaf_method", "leaf_iterations", "reserved0", "reserved1", "reserved2")]
    _fields_ += [("learning_rate", ct.c_float), ("l2_leaf_reg", ct.c_float),
                 ("bias", ct.c_float), ("reserved3", ct.c_uint32)]


class Node(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("feature", "bin", "type", "left", "right", "leaf")]


class StepInfo(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in (
        "completed_iterations", "finished", "node_count", "leaf_count")]
    _fields_ += [("loss", ct.c_float), ("reserved", ct.c_uint32), ("stats", TrainStats)]


@dataclass
class StepResult:
    completed_iterations: int
    finished: bool
    nodes: np.ndarray
    leaf_values: np.ndarray
    leaf_weights: np.ndarray
    loss: float
    stats: dict

    @property
    def iteration(self):
        return self.completed_iterations - 1


@dataclass
class TrainResult:
    trees: tuple
    predictions: np.ndarray
    loss: np.ndarray
    stats: dict

    @property
    def completed_iterations(self):
        return len(self.trees)


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Greedy Metal training requires macOS on Apple Silicon.")
    import fcntl
    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    # Hash just this translation unit and its transitive local headers. Changes
    # to unrelated sessions should not rebuild the greedy runtime.
    pending, sources = [native / "metal_greedy_trainer.mm", native / "metal_sort.mm"], {}
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
    for source, data in sorted(sources.items()):
        digest.update(source.name.encode()); digest.update(data)
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_greedy_{digest.hexdigest()[:20]}.dylib"
    with (build / "greedy-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                completed = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_greedy_trainer.mm"), str(native / "metal_sort.mm"), "-o", str(temporary)],
                    capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError("Could not build greedy Metal runtime:\n" + completed.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    lib = ct.CDLL(str(path))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    text = [ct.c_char_p, ct.c_size_t]
    lib.cbm_greedy_session_create.argtypes = [ct.POINTER(Params), u8, f32, f32, f32,
                                             u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_greedy_session_create_configured.argtypes = [ct.POINTER(Params), ct.POINTER(ObjectiveOptions),
        u8, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_greedy_session_create_query.argtypes = [ct.POINTER(Params), ct.POINTER(ObjectiveOptions),
        ct.POINTER(QueryOptions), u32, ct.c_uint64, u8, f32, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_greedy_session_create_pair.argtypes = [ct.POINTER(Params), ct.POINTER(ObjectiveOptions),
        ct.POINTER(PairOptions), u32, u32, f32, ct.c_uint64, u32, ct.c_uint64,
        u8, f32, f32, u32, u32, u8, ct.POINTER(ct.c_void_p)] + text
    lib.cbm_greedy_session_step.argtypes = [ct.c_void_p, ct.POINTER(StepInfo), ct.POINTER(Node), f32, f32] + text
    lib.cbm_greedy_session_copy_predictions.argtypes = [ct.c_void_p, f32] + text
    lib.cbm_greedy_session_info.argtypes = [ct.c_void_p, ct.POINTER(StepInfo)] + text
    lib.cbm_greedy_session_set_backtracking.argtypes = [ct.c_void_p, ct.c_uint32] + text
    lib.cbm_greedy_session_set_bootstrap.argtypes = [ct.c_void_p, ct.POINTER(BootstrapOptions)] + text
    lib.cbm_greedy_session_set_score_noise.argtypes = [ct.c_void_p, ct.POINTER(ScoreNoiseOptions)] + text
    lib.cbm_greedy_session_set_permutations.argtypes = [ct.c_void_p, ct.c_uint32,
        ct.POINTER(u8), ct.POINTER(f32), f32, u8] + text
    lib.cbm_greedy_session_select_permutation.argtypes = [ct.c_void_p, ct.c_uint32] + text
    lib.cbm_greedy_session_copy_permutation_state.argtypes = [ct.c_void_p, ct.c_uint32, f32, f32, u8] + text
    lib.cbm_greedy_session_close.argtypes = [ct.c_void_p]
    lib.cbm_greedy_session_close.restype = None
    for name in ("create", "create_configured", "create_query", "create_pair", "step", "copy_predictions", "info", "set_backtracking",
                 "set_bootstrap", "set_score_noise", "set_permutations", "select_permutation", "copy_permutation_state"):
        getattr(lib, "cbm_greedy_session_" + name).restype = ct.c_int
    return lib


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or not low <= value <= high:
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


def _check(code, error):
    if code:
        raise RuntimeError("Metal greedy training failed: " + error.value.decode("utf-8", errors="replace"))


def objective_parameter(objective, value=None):
    """Validate the shared scalar objective ABI before constructing a session."""
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {', '.join(OBJECTIVES)}.")
    if value is None and objective in ("Huber", "Expectile", "Lq", "Tweedie"):
        raise ValueError(f"{objective} requires an explicit objective_param.")
    parameter = float(_finite_array("objective_param",
        (.5 if objective in ("LogLinQuantile", "Quantile", "MAE") else 1) if value is None else value, ()))
    if ((objective == "Huber" and parameter < 0) or
            (objective in ("Expectile", "LogLinQuantile", "Quantile") and not 0 <= parameter <= 1) or
            (objective == "Lq" and parameter < 1) or
            (objective == "Tweedie" and not 1 < parameter < 2) or
            (objective == "MAE" and parameter != .5) or
            (objective not in OBJECTIVE_PARAMETERS and objective != "MAE" and parameter != 1)):
        raise ValueError(f"Invalid objective_param for {objective}.")
    return parameter


class TrainingSession:
    """Train one variable-shape tree per step on a persistent Metal dataset.

    ``bins`` is feature-major uint8[features, rows]. Numeric candidates route
    bin > border to the right; one-hot candidates route equality to the right.
    Initial predictions are an external baseline, not stored in returned trees.
    """
    def __init__(self, bins, targets, candidate_features, candidate_bins, *,
                 grow_policy="Lossguide", iterations=100, depth=6, max_leaves=None,
                 min_data_in_leaf=1, learning_rate=.03, l2_leaf_reg=3., bias=0.,
                 score_function="Cosine", objective="RMSE", sample_weight=None,
                 leaf_estimation_iterations=1, leaf_estimation_method="Newton",
                 initial_predictions=None, candidate_types=None, boosting_type="Plain",
                 bootstrap_type="No", leaf_estimation_backtracking="No", random_strength=0.,
                 objective_param=None, random_seed=0, iteration_offset=0,
                 bagging_temperature=1., subsample=1., group_offsets=None, query_beta=1., query_lambda=.01,
                 pair_winners=None, pair_losers=None, pair_weights=None):
        self._handle = ct.c_void_p(); self._lib = None; self._lock = threading.RLock()
        self._steps = []
        self._permutation_count = 1
        self._permutations_configured = False
        policies, objectives = ("Depthwise", "Lossguide", "Region"), OBJECTIVES
        scores, methods = SCORES, ("Newton", "Gradient", "Exact")
        for name, value, supported in (("grow_policy", grow_policy, policies),
                ("objective", objective, objectives), ("score_function", score_function, scores),
                ("leaf_estimation_method", leaf_estimation_method, methods)):
            if value not in supported:
                raise ValueError(f"{name} must be one of {', '.join(supported)}.")
        parameter = objective_parameter(objective, objective_param)
        if leaf_estimation_method == "Newton" and (objective in ("LogLinQuantile", "Quantile", "MAE", "MAPE")
                                                    or (objective == "Lq" and parameter < 2)):
            raise ValueError("Newton is unsupported for this objective or objective parameter.")
        if leaf_estimation_method == "Exact" and objective not in ("Quantile", "MAE", "MAPE"):
            raise ValueError("Exact supports Quantile, MAE and MAPE only.")
        if boosting_type != "Plain":
            raise ValueError("Greedy training currently requires boosting_type=Plain.")
        if bootstrap_type not in BOOTSTRAPS:
            raise ValueError("bootstrap_type must be No, Bayesian, Bernoulli, or Poisson; MVS is unsupported.")
        backtracking = ("No", "AnyImprovement", "Armijo")
        if leaf_estimation_backtracking not in backtracking:
            raise ValueError("leaf_estimation_backtracking must be No, AnyImprovement, or Armijo.")
        strength = _finite_array("random_strength", random_strength, ())
        if strength < 0:
            raise ValueError("random_strength must be finite and nonnegative.")
        iterations = _integer("iterations", iterations, 1, 100000)
        self.random_seed = _integer("random_seed", random_seed, 0, 2**64 - 1)
        self.iteration_offset = _integer("iteration_offset", iteration_offset, 0, 2**32 - 1 - iterations)
        temperature = float(_finite_array("bagging_temperature", bagging_temperature, ()))
        fraction = float(_finite_array("subsample", subsample, ()))
        if temperature < 0:
            raise ValueError("bagging_temperature must be finite and nonnegative.")
        if not 0 < fraction <= 1 or (bootstrap_type == "Poisson" and fraction >= 1):
            raise ValueError("subsample must be in (0, 1], and strictly below 1 for Poisson bootstrap.")
        depth = _integer("depth", depth, 0, {"Depthwise": 16, "Region": 65535, "Lossguide": 2**32 - 1}[grow_policy])
        max_leaves = (depth + 1 if grow_policy == "Region" else
                      (31 if grow_policy == "Lossguide" else 1 << depth)) if max_leaves is None else max_leaves
        max_leaves = _integer("max_leaves", max_leaves, 1, 65536)
        min_data_in_leaf = _integer("min_data_in_leaf", min_data_in_leaf, 1, 16777216)
        leaf_iterations = _integer("leaf_estimation_iterations", leaf_estimation_iterations, 1, 1000)
        bins = np.asarray(bins)
        if bins.ndim != 2 or bins.dtype.kind not in "iu" or not all(bins.shape):
            raise ValueError("bins must be a nonempty feature-major two-dimensional integer array.")
        features, rows = bins.shape
        _integer("rows", rows, 1, 16777216)
        if rows * features > 2**32 - 1 or (bins < 0).any() or (bins > 255).any():
            raise ValueError("bins must lie in [0,255], with at most uint32 cells.")
        bins = np.ascontiguousarray(bins, np.uint8)
        targets = _finite_array("targets", targets, (rows,))
        if objective == "Logloss" and ((targets != 0) & (targets != 1)).any():
            raise ValueError("Logloss targets must be binary 0 or 1 labels.")
        if objective == "CrossEntropy" and ((targets < 0) | (targets > 1)).any():
            raise ValueError("CrossEntropy targets must be in [0, 1].")
        if objective in ("Poisson", "Tweedie") and (targets < 0).any():
            raise ValueError("Poisson/Tweedie targets must be nonnegative.")
        paired = objective == "PairLogit"
        grouped = objective in ("QueryRMSE", "QuerySoftMax")
        if paired:
            from ._query_data import validate_offsets, prepare_pair_arrays
            if sample_weight is not None:
                raise ValueError("PairLogit uses literal incident pair mass; sample_weight must be None.")
            if group_offsets is not None: group_offsets = validate_offsets(group_offsets, rows)
            pair_winners, pair_losers, pair_weights = prepare_pair_arrays(pair_winners, pair_losers, pair_weights, rows, group_offsets)
        elif any(value is not None for value in (pair_winners, pair_losers, pair_weights)):
            raise ValueError("Supplied pair arrays require PairLogit.")
        if grouped:
            from ._query_data import validate_offsets, query_metric
            group_offsets = validate_offsets(group_offsets, rows)
            query_beta = float(_finite_array("query_beta", query_beta, ()))
            query_lambda = float(_finite_array("query_lambda", query_lambda, ()))
        elif (group_offsets is not None and not paired) or query_beta != 1 or query_lambda != .01:
            raise ValueError("Query options require QueryRMSE or QuerySoftMax.")
        weights = np.ones(rows, np.float32) if sample_weight is None else _finite_array("sample_weight", sample_weight, (rows,))
        if (weights < 0).any() or not 0 < weights.sum(dtype=np.float64) < 1e30:
            raise ValueError("sample_weight must be nonnegative with positive total below 1e30.")
        cf, cb = np.asarray(candidate_features), np.asarray(candidate_bins)
        if cf.ndim != 1 or cf.shape != cb.shape:
            raise ValueError("Candidate arrays must be matching vectors.")
        if cf.size and (cf.dtype.kind not in "iu" or cb.dtype.kind not in "iu" or
                       (cf < 0).any() or (cf >= features).any() or (cb < 0).any() or (cb > 255).any()):
            raise ValueError("Invalid candidate feature or bin index.")
        _integer("candidates", cf.size, 0, 2**32 - 1)
        types = np.zeros(cf.size, np.uint8) if candidate_types is None else np.asarray(candidate_types)
        if (types.shape != cf.shape or types.dtype.kind not in "biu" or
                ((types != 0) & (types != 1)).any()):
            raise ValueError("candidate_types must contain numeric(0)/one-hot(1) flags.")
        if ((types == 0) & (cb == 255)).any():
            raise ValueError("Numeric candidate borders must be below 255.")
        type_by_feature = {}
        for feature, kind in zip(cf, types):
            if type_by_feature.setdefault(int(feature), int(kind)) != kind:
                raise ValueError("A feature cannot mix numeric and one-hot candidates.")
        cf, cb, types = (np.ascontiguousarray(cf, np.uint32), np.ascontiguousarray(cb, np.uint32),
                         np.ascontiguousarray(types, np.uint8))
        bins_per_feature = int(bins.max()) + 1
        if cb.size:
            bins_per_feature = max(bins_per_feature, int((cb + np.where(types, 1, 2)).max()))
        scalars = _finite_array("training scalars", [learning_rate, l2_leaf_reg, bias], (3,))
        if not 0 < scalars[0] <= 1 or scalars[1] < 0:
            raise ValueError("learning_rate must be in (0,1] and l2_leaf_reg nonnegative.")
        initial = None if initial_predictions is None else _finite_array("initial_predictions", initial_predictions, (rows,))
        self._params = Params(rows, features, len(cf), bins_per_feature, iterations, depth,
            max_leaves, min_data_in_leaf, policies.index(grow_policy), objectives.index(objective),
            scores.index(score_function), methods.index(leaf_estimation_method), leaf_iterations,
            0, 0, 0, *map(float, scalars), 0)
        self.objective, self.grow_policy, self.objective_param = objective, grow_policy, parameter
        self._objective_options = ObjectiveOptions(objectives.index(objective),
            methods.index(leaf_estimation_method), parameter, 0)
        self._bootstrap_options = BootstrapOptions(BOOTSTRAPS.index(bootstrap_type),
            self.random_seed & 0xffffffff, self.random_seed >> 32, self.iteration_offset,
            temperature, fraction, 0., 0, 0., 0, 0, 0)
        self._noise_options = ScoreNoiseOptions(float(strength), 0, 0, 0)
        if grouped:
            query_metric(np.zeros(rows), targets, weights, group_offsets, objective, query_beta, query_lambda)
        self._lib = _load(build_library())
        error = ct.create_string_buffer(2048)
        try:
            create = self._lib.cbm_greedy_session_create_configured
            arguments = [ct.byref(self._params), ct.byref(self._objective_options)]
            if grouped:
                create = self._lib.cbm_greedy_session_create_query
                query = QueryOptions(len(group_offsets) - 1, query_beta, query_lambda, 0)
                arguments += [ct.byref(query), _u32(group_offsets), len(group_offsets)]
            if paired:
                pair = PairOptions(len(pair_winners), 0 if group_offsets is None else len(group_offsets)-1, 0, 0)
                _check(self._lib.cbm_greedy_session_create_pair(*arguments, ct.byref(pair), _u32(pair_winners),
                    _u32(pair_losers), _f32(pair_weights), len(pair_winners),
                    None if group_offsets is None else _u32(group_offsets), 0 if group_offsets is None else len(group_offsets),
                    _u8(bins), _f32(targets), None if initial is None else _f32(initial), _u32(cf), _u32(cb), _u8(types),
                    ct.byref(self._handle), error, len(error)), error)
            else:
                _check(create(*arguments, _u8(bins), _f32(targets),
                    _f32(weights), None if initial is None else _f32(initial), _u32(cf), _u32(cb), _u8(types),
                    ct.byref(self._handle), error, len(error)), error)
            _check(self._lib.cbm_greedy_session_set_backtracking(self._handle,
                backtracking.index(leaf_estimation_backtracking), error, len(error)), error)
            _check(self._lib.cbm_greedy_session_set_bootstrap(self._handle,
                ct.byref(self._bootstrap_options), error, len(error)), error)
            _check(self._lib.cbm_greedy_session_set_score_noise(self._handle,
                ct.byref(self._noise_options), error, len(error)), error)
            self._initial_loss = float(self._info().loss)
        except Exception:
            self.close()
            raise

    def _require_open(self):
        if not self._handle.value:
            raise RuntimeError("Greedy training session is closed.")

    def _info(self):
        self._require_open()
        info, error = StepInfo(), ct.create_string_buffer(2048)
        _check(self._lib.cbm_greedy_session_info(self._handle, ct.byref(info), error, len(error)), error)
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
            return {"iteration_offset": self.iteration_offset + self.completed_iterations, "mvs_lambda": None}

    def predictions(self):
        with self._lock:
            self._require_open()
            output = np.empty(self._params.rows, np.float32)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_greedy_session_copy_predictions(self._handle, _f32(output), error, len(error)), error)
            return output

    def configure_permutations(self, bins_list, initial_predictions=None):
        with self._lock:
            self._require_open()
            if self.completed_iterations or self._permutations_configured:
                raise ValueError("Permutations can be configured once before training.")
            p = self._params
            matrices = np.asarray(bins_list)
            if (matrices.ndim != 3 or not 1 <= matrices.shape[0] <= 64
                    or matrices.shape[1:] != (p.features, p.rows) or matrices.dtype.kind not in "iu"
                    or (matrices < 0).any() or (matrices >= p.bins_per_feature).any()):
                raise ValueError("Permutation bins must have shape (1..64, features, rows) within the original bin range.")
            matrices = np.ascontiguousarray(matrices, np.uint8)
            count = len(matrices)
            cursors = None if initial_predictions is None else _finite_array(
                "Permutation raw cursors", initial_predictions, (count, p.rows))
            pointers = (ct.POINTER(ct.c_uint8) * count)(*(_u8(matrix) for matrix in matrices))
            cursor_pointers = None if cursors is None else (ct.POINTER(ct.c_float) * count)(*(_f32(c) for c in cursors))
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_greedy_session_set_permutations(self._handle, count, pointers,
                cursor_pointers, None, None, error, len(error)), error)
            self._permutation_count = count
            self._permutations_configured = True
            self._initial_loss = float(self._info().loss)

    def select_permutation(self, index):
        with self._lock:
            self._require_open()
            index = _integer("Search permutation index", index, 0, self._permutation_count - 1)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_greedy_session_select_permutation(self._handle, index, error, len(error)), error)

    @property
    def permutation_state(self):
        with self._lock:
            self._require_open()
            count = self._permutation_count
            cursors = np.empty((count, self._params.rows), np.float32)
            lambdas, valid = np.empty(count, np.float32), np.empty(count, np.uint8)
            error = ct.create_string_buffer(2048)
            _check(self._lib.cbm_greedy_session_copy_permutation_state(self._handle, count,
                _f32(cursors), _f32(lambdas), _u8(valid), error, len(error)), error)
            return {"predictions": cursors, "mvs_lambdas": lambdas, "mvs_valid": valid}

    def step(self):
        with self._lock:
            self._require_open()
            if self.completed_iterations >= self._params.iterations:
                raise RuntimeError("Greedy session has no remaining iterations.")
            capacity = self._params.max_leaves
            nodes = np.zeros((2 * capacity - 1, 6), np.uint32)
            values, weights = np.zeros(capacity, np.float32), np.zeros(capacity, np.float32)
            info, error = StepInfo(), ct.create_string_buffer(2048)
            _check(self._lib.cbm_greedy_session_step(self._handle, ct.byref(info),
                nodes.ctypes.data_as(ct.POINTER(Node)), _f32(values), _f32(weights), error, len(error)), error)
            if not (1 <= info.node_count <= len(nodes) and 1 <= info.leaf_count <= capacity):
                raise RuntimeError("Metal returned invalid greedy tree output sizes.")
            result = StepResult(int(info.completed_iterations), bool(info.finished),
                nodes[:info.node_count].copy(), values[:info.leaf_count].copy(),
                weights[:info.leaf_count].copy(), float(info.loss), _stats(info.stats))
            self._steps.append(result)
            return result

    def result(self):
        with self._lock:
            self._require_open()
            result = TrainResult(tuple(self._steps), self.predictions(),
                np.array([self._initial_loss, *[step.loss for step in self._steps]], np.float32),
                _stats(self._info().stats))
            result.stats["bootstrap_state"] = self.bootstrap_state
            return result

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is not None:
            with lock:
                if self._handle.value and self._lib is not None:
                    self._lib.cbm_greedy_session_close(self._handle)
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


Session = TrainingSession


def train(bins, targets, candidate_features, candidate_bins, **options):
    with TrainingSession(bins, targets, candidate_features, candidate_bins, **options) as session:
        while session.completed_iterations < session._params.iterations:
            session.step()
        return session.result()
