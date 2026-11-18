"""Exact CatBoost compound projection hashes and stable 64-bit Metal grouping.

This primitive supplies tensor identities and history segments. The trainer is
responsible for generating legal CUDA projections and scheduling their CTRs.
"""

import ctypes as ct
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np


class _Params(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("rows", "cat_features", "bin_features", "components")]


class _Stats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("device_name", ct.c_char * 256)]


def _build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal projection kernels require macOS on Apple Silicon.")
    import fcntl

    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_projection.h", "metal_projection.mm", "metal_projection_kernels.h",
                 "metal_sort.h", "metal_sort.mm", "metal_sort_kernels.h"):
        digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_projection_{digest.hexdigest()[:20]}.dylib"
    with (build / "projection-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                process = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_projection.mm"), str(native / "metal_sort.mm"),
                    "-o", str(temporary),
                ], capture_output=True, text=True, check=False)
                if process.returncode:
                    raise RuntimeError("Could not build Metal projection runtime:\n" + process.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u8, u32, u64 = ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_uint64)
    library.cbm_projection_group.argtypes = [ct.POINTER(_Params), u32, u8, u8, u32, u32, u32,
                                            u64, u64, u32, ct.POINTER(_Stats), ct.c_char_p, ct.c_size_t]
    library.cbm_projection_group.restype = ct.c_int
    return library


def _unsigned(values, dtype, name, ndim):
    values = np.asarray(values)
    if (values.ndim != ndim or (values.size and values.dtype.kind not in "iu")
            or (values < 0).any() or (values > np.iinfo(dtype).max).any()):
        raise ValueError(f"{name} must contain unsigned {np.dtype(dtype).itemsize * 8}-bit integers in {ndim} dimensions.")
    return np.ascontiguousarray(values, dtype=dtype)


@dataclass
class ProjectionGroups:
    row_hashes: np.ndarray
    sorted_hashes: np.ndarray
    sorted_rows: np.ndarray
    unique_hashes: np.ndarray
    category_bins: np.ndarray
    row_bins: np.ndarray
    stats: dict


def group_projection(cat_hashes, bins=None, *, component_types=None,
                     component_features=None, component_thresholds=None, permutation=None):
    """Hash and group one projection, retaining exclusive-history row order.

    Matrices are feature-major: ``(features, rows)``. Components have types
    0=original categorical hash, 1=quantized bin greater than threshold, and
    2=quantized bin equal to threshold. Types must follow that order, matching
    the model's categorical/numeric/one-hot projection lists. A category hash
    is sign-extended from int32 before CalcHash; predicates contribute 0 or 1.

    By default every supplied categorical feature forms the projection. Empty
    categorical matrices allow projections composed solely of predicates.
    """
    cats = _unsigned(cat_hashes, np.uint32, "cat_hashes", 2)
    rows = cats.shape[1]
    if rows > (1 << 24):
        raise ValueError("Projection rows cannot exceed 16777216.")
    bins = np.empty((0, rows), np.uint8) if bins is None else _unsigned(bins, np.uint8, "bins", 2)
    if bins.shape[1] != rows:
        raise ValueError("Categorical and bin matrices must have the same row count.")
    types = (np.zeros(len(cats), np.uint8) if component_types is None else
             _unsigned(component_types, np.uint8, "component_types", 1))
    features = (np.arange(len(cats), dtype=np.uint32) if component_features is None else
                _unsigned(component_features, np.uint32, "component_features", 1))
    thresholds = (np.zeros(len(types), np.uint32) if component_thresholds is None else
                  _unsigned(component_thresholds, np.uint32, "component_thresholds", 1))
    if not len(types) or features.shape != types.shape or thresholds.shape != types.shape:
        raise ValueError("Projection requires equally sized nonempty component vectors.")
    if (types > 2).any() or (types[1:] < types[:-1]).any():
        raise ValueError("Projection components must be ordered categorical, numeric, then one-hot.")
    if ((features[types == 0] >= len(cats)).any() or (features[types != 0] >= len(bins)).any()
            or (thresholds[types != 0] > 255).any()):
        raise ValueError("Projection component feature or threshold is out of range.")
    if permutation is not None:
        permutation = _unsigned(permutation, np.uint32, "permutation", 1)
        if (permutation.shape != (rows,) or (permutation >= rows).any()
                or not np.all(np.bincount(permutation, minlength=rows) == 1)):
            raise ValueError("Projection permutation must contain each row exactly once.")
    row_hashes, sorted_hashes = np.empty(rows, np.uint64), np.empty(rows, np.uint64)
    sorted_rows = np.empty(rows, np.uint32)
    params = _Params(rows, len(cats), len(bins), len(types))
    stats, error = _Stats(), ct.create_string_buffer(2048)
    u8 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint8))
    u32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint32))
    u64 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint64))
    code = _load(_build_library()).cbm_projection_group(ct.byref(params), u32(cats), u8(bins),
        u8(types), u32(features), u32(thresholds), None if permutation is None else u32(permutation),
        u64(row_hashes), u64(sorted_hashes), u32(sorted_rows), ct.byref(stats), error, len(error))
    if code:
        raise RuntimeError("Metal projection grouping failed: " + error.value.decode("utf-8", errors="replace"))
    boundaries = np.empty(rows, bool)
    if rows:
        boundaries[0] = True
        boundaries[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    categories = np.cumsum(boundaries, dtype=np.uint32) - np.uint32(1)
    row_bins = np.empty(rows, np.uint32)
    row_bins[sorted_rows] = categories
    return ProjectionGroups(row_hashes, sorted_hashes, sorted_rows, sorted_hashes[boundaries], categories,
        row_bins, {"backend": "Metal", "device": stats.device_name.decode("utf-8"), "hash_bits": 64,
                   "kernel_dispatches": int(stats.kernel_dispatches), "gpu_seconds": stats.gpu_seconds,
                   "radix_passes": 16 if rows else 0})
