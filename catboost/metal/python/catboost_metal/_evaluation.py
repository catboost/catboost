"""Resident Metal prediction cursors for per-iteration training evaluation.

This deliberately uses the trainer's sequential float32 additions. General
model inference uses its separate compensated accumulator and float64 outputs.
"""

import ctypes as ct
import functools
import hashlib
import numbers
from pathlib import Path
import platform
import subprocess
import threading

import numpy as np


class EvaluationParams(ct.Structure):
    _fields_ = [("rows", ct.c_uint32), ("features", ct.c_uint32),
                ("max_depth", ct.c_uint32), ("bias", ct.c_float)]


class EvaluationMatrixParams(ct.Structure):
    _fields_ = [("rows", ct.c_uint32), ("features", ct.c_uint32),
                ("max_depth", ct.c_uint32), ("classes", ct.c_uint32)]


class EvaluationStats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("dataset_uploads", ct.c_uint64), ("bins_upload_bytes", ct.c_uint64),
                ("resident_bytes", ct.c_uint64), ("tree_upload_bytes", ct.c_uint64),
                ("device_name", ct.c_char * 256)]


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Resident Metal evaluation requires macOS on Apple Silicon.")
    import fcntl

    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_evaluation.h", "metal_evaluation.mm"):
        digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_evaluation_{digest.hexdigest()[:20]}.dylib"
    with (build / "evaluation_build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                result = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_evaluation.mm"), "-o", str(temporary),
                ], capture_output=True, text=True, check=False)
                if result.returncode:
                    raise RuntimeError("Could not build Metal evaluation:\n" + result.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u8, u32, f32 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    tail = [ct.c_char_p, ct.c_size_t]
    library.cbm_evaluation_create.argtypes = [ct.POINTER(EvaluationParams), u8, ct.c_uint64,
        f32, ct.c_uint64, ct.POINTER(ct.c_void_p)] + tail
    library.cbm_evaluation_create.restype = ct.c_int
    library.cbm_evaluation_create_matrix.argtypes = [ct.POINTER(EvaluationMatrixParams),
        u8, ct.c_uint64, f32, ct.c_uint64, f32, ct.c_uint64, ct.POINTER(ct.c_void_p)] + tail
    library.cbm_evaluation_create_matrix.restype = ct.c_int
    library.cbm_evaluation_add_tree.argtypes = [ct.c_void_p, ct.c_uint32,
        u32, ct.c_uint64, u32, ct.c_uint64, u8, ct.c_uint64, f32, ct.c_uint64,
        f32, ct.c_uint64] + tail
    library.cbm_evaluation_add_tree.restype = ct.c_int
    library.cbm_evaluation_add_tree_matrix.argtypes = library.cbm_evaluation_add_tree.argtypes
    library.cbm_evaluation_add_tree_matrix.restype = ct.c_int
    library.cbm_evaluation_predictions.argtypes = [ct.c_void_p, f32, ct.c_uint64] + tail
    library.cbm_evaluation_predictions.restype = ct.c_int
    library.cbm_evaluation_predictions_matrix.argtypes = library.cbm_evaluation_predictions.argtypes
    library.cbm_evaluation_predictions_matrix.restype = ct.c_int
    library.cbm_evaluation_stats.argtypes = [ct.c_void_p, ct.POINTER(EvaluationStats)] + tail
    library.cbm_evaluation_stats.restype = ct.c_int
    library.cbm_evaluation_destroy.argtypes = [ct.c_void_p]
    library.cbm_evaluation_destroy.restype = None
    return library


def _integer(value, name, minimum, maximum):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
            or not minimum <= value <= maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}].")
    return int(value)


def _array(value, name, dimensions, kinds):
    try:
        value = (np.asarray(value, dtype=np.uint32)
                 if kinds == "iu" and isinstance(value, (list, tuple)) and not value
                 else np.asarray(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array.") from error
    if value.ndim != dimensions or value.dtype.kind not in kinds:
        raise ValueError(f"{name} must be a {dimensions}-dimensional "
                         f"{'integer' if kinds == 'iu' else 'real numeric'} array.")
    return value


def _float_array(value, name, count, *, classes=1):
    value = _array(value, name, 1 if classes == 1 else 2, "iuf")
    shape = (count,) if classes == 1 else (count, classes)
    if value.shape != shape:
        raise ValueError(f"{name} must have exactly shape {shape}.")
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite within the float32 range.")
    return value


def _float_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be finite within the float32 range.")
    try:
        value = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite within the float32 range.") from error
    if not np.isfinite(value) or abs(value) > float(np.finfo(np.float32).max):
        raise ValueError(f"{name} must be finite within the float32 range.")
    return value


def _pointer(array, dtype):
    return array.ctypes.data_as(ct.POINTER(dtype))


class EvaluationCursor:
    """Own one quantized validation dataset and its running Metal predictions.

    ``bins`` has integer shape [features, rows] with values in [0, 255].
    ``classes=1`` retains scalar predictions [rows] and leaves [2**depth]. For
    ``classes>1``, predictions have row-major shape [rows,classes], leaves have
    shape [2**depth,classes], and bias is a scalar or a vector [classes]. One
    bins matrix is shared by every output, and one GPU dispatch updates all
    outputs per tree. ``initial_predictions`` has the prediction shape and
    replaces bias when supplied. Tree arrays contain exactly their active
    depth; leaf values are already scaled by learning rate. Type 0 is numeric ``bin >
    border`` (border <=254); type 1 is one-hot ``bin == category`` (<=255).

    Predictions are independent float32 copies. Reading them performs no GPU
    work or dataset upload. Statistics remain readable after idempotent close;
    other operations on a closed cursor raise RuntimeError. The native session
    enforces a 1 GiB total resident buffer limit, depths from zero to 16, and
    from one to 64 output columns. Float32 overflow bounds apply per column.
    """

    def __init__(self, bins, *, bias=0.0, max_depth=8, initial_predictions=None, classes=1):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self._library = None
        self._final_stats = None
        bins = _array(bins, "bins", 2, "iu")
        self._features, self._rows = bins.shape
        self._max_depth = _integer(max_depth, "max_depth", 0, 16)
        self._classes = _integer(classes, "classes", 1, 64)
        self._prediction_shape = (self._rows,) if self._classes == 1 else (self._rows, self._classes)
        if (self._features > 2**32 - 1 or self._rows > 1 << 27 or bins.size > 2**32 - 1
                or self._rows * self._classes > 2**32 - 1):
            raise ValueError("Evaluation data exceeds the GPU row or index limit.")
        resident_bytes = (max(1, bins.size) + max(1, self._rows * self._classes * 4)
                          + 2 * max(1, self._max_depth * 4) + max(1, self._max_depth)
                          + (1 << self._max_depth) * self._classes * 4)
        if resident_bytes > 1 << 30:
            raise ValueError("Evaluation cursor exceeds the 1 GiB resident memory limit.")
        if (bins < 0).any() or (bins > 255).any():
            raise ValueError("bins values must be in [0, 255].")
        if self._classes == 1 or isinstance(bias, numbers.Real):
            bias = _float_scalar(bias, "bias")
            if self._classes > 1:
                bias = np.full(self._classes, bias, np.float32)
        else:
            bias = _float_array(bias, "bias", self._classes)
        initial = (None if initial_predictions is None else
                   _float_array(initial_predictions, "initial_predictions", self._rows,
                                classes=self._classes))
        bins = np.ascontiguousarray(bins, dtype=np.uint8)
        self._library = _load(build_library())
        if self._classes == 1:
            params = EvaluationParams(self._rows, self._features, self._max_depth, bias)
            self._call(self._library.cbm_evaluation_create, ct.byref(params),
                       _pointer(bins, ct.c_uint8), bins.size,
                       None if initial is None else _pointer(initial, ct.c_float),
                       0 if initial is None else initial.size, ct.byref(self._handle))
        else:
            params = EvaluationMatrixParams(self._rows, self._features, self._max_depth, self._classes)
            self._call(self._library.cbm_evaluation_create_matrix, ct.byref(params),
                       _pointer(bins, ct.c_uint8), bins.size, _pointer(bias, ct.c_float), bias.size,
                       None if initial is None else _pointer(initial, ct.c_float),
                       0 if initial is None else initial.size, ct.byref(self._handle))

    @staticmethod
    def _call(function, *arguments):
        error = ct.create_string_buffer(2048)
        if function(*arguments, error, len(error)):
            raise RuntimeError("Metal evaluation failed: " + error.value.decode("utf-8", errors="replace"))

    def _require_open(self):
        if not self._handle.value:
            raise RuntimeError("Evaluation cursor is closed.")

    def add_tree(self, depth, split_features, split_bins, leaf_values, *, split_types=None):
        """Upload one tree, advance the resident cursor, and copy raw predictions."""
        with self._lock:
            self._require_open()
            depth = _integer(depth, "depth", 0, self._max_depth)
            features = _array(split_features, "split_features", 1, "iu")
            borders = _array(split_bins, "split_bins", 1, "iu")
            types = (np.zeros(depth, np.uint8) if split_types is None else
                     _array(split_types, "split_types", 1, "iu"))
            if any(array.size != depth for array in (features, borders, types)):
                raise ValueError("Split arrays must have exactly depth elements.")
            if (features < 0).any() or (features >= self._features).any():
                raise ValueError("An active split feature is outside the input feature count.")
            if (types < 0).any() or (types > 1).any():
                raise ValueError("split_types values must be 0 (numeric) or 1 (one-hot).")
            if ((borders < 0).any() or (borders > 255).any()
                    or ((types == 0) & (borders >= 255)).any()):
                raise ValueError("Split bins must be <=254 for numeric or <=255 for one-hot splits.")
            leaves = _float_array(leaf_values, "leaf_values", 1 << depth, classes=self._classes)
            features = np.ascontiguousarray(features, dtype=np.uint32)
            borders = np.ascontiguousarray(borders, dtype=np.uint32)
            types = np.ascontiguousarray(types, dtype=np.uint8)
            predictions = np.empty(self._prediction_shape, np.float32)
            add_tree = (self._library.cbm_evaluation_add_tree if self._classes == 1 else
                        self._library.cbm_evaluation_add_tree_matrix)
            self._call(add_tree, self._handle, depth,
                       _pointer(features, ct.c_uint32), features.size,
                       _pointer(borders, ct.c_uint32), borders.size,
                       _pointer(types, ct.c_uint8), types.size,
                       _pointer(leaves, ct.c_float), leaves.size,
                       _pointer(predictions, ct.c_float), predictions.size)
            return predictions

    def predictions(self):
        """Copy current raw predictions without uploading data or dispatching."""
        with self._lock:
            self._require_open()
            result = np.empty(self._prediction_shape, np.float32)
            predictions = (self._library.cbm_evaluation_predictions if self._classes == 1 else
                           self._library.cbm_evaluation_predictions_matrix)
            self._call(predictions, self._handle,
                       _pointer(result, ct.c_float), result.size)
            return result

    @property
    def stats(self):
        with self._lock:
            if self._final_stats is not None:
                return self._final_stats.copy()
            self._require_open()
            stats = EvaluationStats()
            self._call(self._library.cbm_evaluation_stats, self._handle, ct.byref(stats))
            return {"backend": "Metal", "device": stats.device_name.decode("utf-8"),
                    "kernel_dispatches": int(stats.kernel_dispatches),
                    "gpu_seconds": float(stats.gpu_seconds),
                    "dataset_uploads": int(stats.dataset_uploads),
                    "bins_upload_bytes": int(stats.bins_upload_bytes),
                    "resident_bytes": int(stats.resident_bytes),
                    "tree_upload_bytes": int(stats.tree_upload_bytes),
                    "output_dimensions": self._classes,
                    "accumulation": "sequential float32"}

    def close(self):
        """Release all owned GPU buffers; repeated calls are harmless."""
        with self._lock:
            if self._handle.value:
                try:
                    self._final_stats = self.stats
                finally:
                    self._library.cbm_evaluation_destroy(self._handle)
                    self._handle = ct.c_void_p()

    def __enter__(self):
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        # Construction can fail before a native session exists. Interpreter
        # teardown must never turn resource cleanup into an unraisable error.
        try:
            self.close()
        except Exception:
            pass
