"""Actual Metal vector-regression derivatives and leaf directions from CUDA."""
import ctypes as ct
import functools
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np

from ._multiclass import _integer, _finite_array, _weights, _method
from ._native import TrainStats, _f32, _u32, _stats


def build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Multioutput Metal math requires macOS on Apple Silicon.")
    import fcntl
    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_multioutput_math.mm", "metal_multioutput_math.h",
                 "metal_multioutput_math_kernels.h", "metal_trainer.h", "metal_multiclass_scores.h"):
        digest.update(name.encode()); digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_multioutput_math_{digest.hexdigest()[:20]}.dylib"
    with (build / "multioutput-math-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                result = subprocess.run(["xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc",
                    "-dynamiclib", "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_multioutput_math.mm"), "-o", str(temporary)], capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError("Could not build multioutput Metal math:\n" + result.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@functools.lru_cache(maxsize=4)
def _load(path):
    lib = ct.CDLL(str(path))
    f32, u32 = ct.POINTER(ct.c_float), ct.POINTER(ct.c_uint32)
    lib.cbm_multioutput_math.argtypes = [ct.c_uint32] * 5 + [ct.c_float,
        f32, f32, f32, u32, f32, f32, f32, f32, ct.POINTER(TrainStats), ct.c_char_p, ct.c_size_t]
    lib.cbm_multioutput_math.restype = ct.c_int
    return lib


def objective_and_leaf_directions(predictions, targets, *, objective="MultiRMSE",
                                  leaf_ids=None, leaves=None, sample_weight=None,
                                  l2_leaf_reg=3., leaf_estimation_method="Newton"):
    """Evaluate CUDA objective equations and estimate fixed-partition leaves on GPU.

    Predictions are dimension-major [D,N]. MultiRMSE targets have [D,N];
    uncertainty targets have [N], with predictions (mean, log standard deviation).
    The returned loss is weighted SSE per row for MultiRMSE or Gaussian NLL.
    The reported metric is sqrt(sum(SSE)/sum(weights)) or mean NLL respectively.
    """
    names = ("MultiRMSE", "RMSEWithUncertainty", "MultiLogloss", "MultiCrossEntropy")
    if objective not in names:
        raise ValueError("Unsupported multioutput objective.")
    objective_id = names.index(objective)
    predictions = np.asarray(predictions)
    if predictions.ndim != 2:
        raise ValueError("predictions must have dimension-major shape [D,rows].")
    dimensions = _integer("dimensions", predictions.shape[0], 2, 64)
    rows = _integer("rows", predictions.shape[1], 1, 16777216)
    if objective_id == 1 and dimensions != 2:
        raise ValueError("RMSEWithUncertainty requires two prediction dimensions.")
    predictions = _finite_array("predictions", predictions, (dimensions, rows))
    targets = _finite_array("targets", targets, (rows,) if objective_id == 1 else (dimensions, rows))
    if objective_id >= 2 and ((targets < 0).any() or (targets > 1).any()):
        raise ValueError("Multilabel targets must be in [0,1].")
    if objective_id == 2 and ((targets != 0) & (targets != 1)).any():
        raise ValueError("MultiLogloss targets must be binary.")
    weights = _weights(sample_weight, rows)
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
    method = _method(leaf_estimation_method)
    memory = 4 * (rows * ((1 if objective_id == 1 else dimensions) + 3 * dimensions + 3)
                  + leaves * (3 * dimensions + 3) + 2)
    if memory > 1 << 30:
        raise ValueError("Multioutput math working set exceeds 1 GiB.")
    gradients, hessian = np.empty_like(predictions), np.empty_like(predictions)
    losses = np.empty(rows, np.float32)
    directions = np.empty((leaves, dimensions), np.float32)
    error, stats = ct.create_string_buffer(2048), TrainStats()
    lib = _load(build_library())
    code = lib.cbm_multioutput_math(rows, dimensions, objective_id, leaves, method, float(l2),
        _f32(targets), _f32(weights), _f32(predictions), _u32(leaf_ids), _f32(gradients),
        _f32(hessian), _f32(losses), _f32(directions), ct.byref(stats), error, len(error))
    if code:
        raise RuntimeError("Metal multioutput math failed: " + error.value.decode("utf-8", errors="replace"))
    metric = losses.sum(dtype=np.float64) / weights.sum(dtype=np.float64)
    if objective_id == 0:
        metric = np.sqrt(metric)
    return dict(gradients=gradients, hessian_diagonal=hessian, weighted_losses=losses,
                directions=directions, metric=float(metric), stats=_stats(stats))
