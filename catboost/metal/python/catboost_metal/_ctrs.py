"""Metal category-statistic primitives with CUDA's exclusive-history semantics.

Metal stably sorts category hashes in permutation order and performs segmented
scans, prior application, scatter, and final category reductions. Target-based
training encodings exclude the current/future rows; inference uses full learn
statistics. CUDA BuildCtrTarget uses unit row weights for these statistics,
independently of the sample weights used to train the trees.

These primitives are single categorical projections, not yet the complete
permutation-dataset or tree-dependent CTR feature scheduler.
"""

import ctypes as ct
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import numbers
from pathlib import Path
import platform
import subprocess

import numpy as np

from ._sort import stable_sort


CTR_TYPES = {"Borders": 0, "Buckets": 1, "FloatTargetMeanValue": 2, "FeatureFreq": 3}


class _Params(ct.Structure):
    _fields_ = [("rows", ct.c_uint32), ("categories", ct.c_uint32),
                ("ctr_type", ct.c_uint32), ("target_border", ct.c_uint32),
                ("prior_numerator", ct.c_float), ("prior_denominator", ct.c_float)]


class _Stats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("device_name", ct.c_char * 256)]


def _build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal CTR kernels require macOS on Apple Silicon.")
    import fcntl

    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_ctrs.h", "metal_ctrs.mm", "metal_ctr_kernels.h"):
        digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_ctrs_{digest.hexdigest()[:20]}.dylib"
    with (build / "ctr-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                process = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_ctrs.mm"), "-o", str(temporary),
                ], capture_output=True, text=True, check=False)
                if process.returncode:
                    raise RuntimeError("Could not build Metal CTR runtime:\n" + process.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u32, f32 = ct.POINTER(ct.c_uint32), ct.POINTER(ct.c_float)
    library.cbm_compute_ctrs.argtypes = [ct.POINTER(_Params), u32, u32, f32, f32,
                                       f32, u32, ct.POINTER(_Stats), ct.c_char_p, ct.c_size_t]
    library.cbm_compute_ctrs.restype = ct.c_int
    library.cbm_compute_ctrs_grouped.argtypes = [ct.POINTER(_Params), u32, u32, f32, u32,
                                               f32, f32, u32, ct.POINTER(_Stats), ct.c_char_p, ct.c_size_t]
    library.cbm_compute_ctrs_grouped.restype = ct.c_int
    return library


def _hashes(values):
    values = np.asarray(values)
    if (values.ndim != 1 or (values.size and values.dtype.kind not in "iu")
            or (values < 0).any() or (values > np.iinfo(np.uint32).max).any()):
        raise ValueError("category_hashes must be a vector of unsigned 32-bit CatBoost hashes.")
    return np.ascontiguousarray(values, dtype=np.uint32)


@dataclass
class CtrResult:
    hashes: np.ndarray
    sums: np.ndarray
    counts: np.ndarray
    values: np.ndarray
    ctr_type: str
    target_border_idx: int
    prior_numerator: float
    prior_denominator: float
    stats: dict

    def full_values(self, category_hashes):
        """Encode held-out rows using learn-only tables, never held-out targets."""
        values = _hashes(category_hashes)
        positions = np.searchsorted(self.hashes, values)
        found = positions < len(self.hashes)
        found[found] &= self.hashes[positions[found]] == values[found]
        sums = np.zeros(len(values), np.float32)
        counts = np.zeros(len(values), np.float32)
        sums[found] = self.sums[positions[found]]
        counts[found] = self.counts[positions[found]]
        if self.ctr_type == "FeatureFreq":
            numerator = counts
            denominator = np.float32(self.counts.sum(dtype=np.uint64))
        else:
            numerator, denominator = sums, counts
        return np.asarray((numerator + np.float32(self.prior_numerator))
                          / (denominator + np.float32(self.prior_denominator)), dtype=np.float32)


def compute_ctr(category_hashes, targets=None, *, ctr_type="Borders", permutation=None,
                target_border_idx=0, prior_numerator=0.5, prior_denominator=1.0, group_ids=None):
    """Compute one CUDA CTR configuration on Metal, in original row order.

    Borders and Buckets consume already binarized targets. FloatTargetMeanValue
    consumes float targets. FeatureFreq ignores targets and uses learn-only
    frequencies. The caller supplies the desired row permutation; absent one,
    original row order is the history order. Optional unsigned group_ids are
    in original row order; each group must be contiguous in the permutation.
    Target-based histories then exclude every row of the current group.
    """
    hashes = _hashes(category_hashes)
    rows = len(hashes)
    if not 1 <= rows <= (1 << 24):
        raise ValueError("CTR rows must be in [1, 16777216].")
    if ctr_type not in CTR_TYPES:
        raise ValueError("CTR type must be Borders, Buckets, FloatTargetMeanValue, or FeatureFreq.")
    if (isinstance(target_border_idx, bool) or not isinstance(target_border_idx, numbers.Integral)
            or not 0 <= target_border_idx <= 255):
        raise ValueError("CTR target_border_idx must be an integer in [0, 255].")
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
           for value in (prior_numerator, prior_denominator)):
        raise ValueError("CTR priors must be finite float32 scalars.")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            numerator, denominator = np.float32(prior_numerator), np.float32(prior_denominator)
    except (TypeError, ValueError) as exc:
        raise ValueError("CTR priors must be finite float32 scalars.") from exc
    if (np.ndim(numerator) or np.ndim(denominator) or not np.isfinite([numerator, denominator]).all()
            or denominator <= 0):
        raise ValueError("CTR priors must be finite, with positive prior_denominator.")
    if targets is None and ctr_type == "FeatureFreq":
        targets = np.zeros(rows, np.float32)
    raw_targets = np.asarray(targets)
    if raw_targets.shape != (rows,) or raw_targets.dtype.kind not in "biuf":
        raise ValueError("CTR targets must be a numeric vector matching category_hashes.")
    with np.errstate(over="ignore", invalid="ignore"):
        targets = np.ascontiguousarray(raw_targets, dtype=np.float32)
    if not np.isfinite(targets).all():
        raise ValueError("CTR targets must contain finite float32 values.")
    if ctr_type in ("Borders", "Buckets") and (
            (targets < 0).any() or (targets > 255).any() or (targets != np.floor(targets)).any()):
        raise ValueError("Borders/Buckets CTR targets must be bins in [0, 255].")
    if permutation is None:
        permutation = np.arange(rows, dtype=np.uint32)
    else:
        permutation = np.asarray(permutation)
        if (permutation.shape != (rows,) or permutation.dtype.kind not in "iu"
                or (permutation < 0).any() or (permutation >= rows).any()):
            raise ValueError("CTR permutation must contain every row index exactly once.")
        permutation = np.ascontiguousarray(permutation, dtype=np.uint32)
        if not np.all(np.bincount(permutation, minlength=rows) == 1):
            raise ValueError("CTR permutation must contain every row index exactly once.")
    if group_ids is not None:
        group_ids = np.asarray(group_ids)
        if (group_ids.shape != (rows,) or group_ids.dtype.kind not in "iu"
                or (group_ids < 0).any() or (group_ids > np.iinfo(np.uint32).max).any()):
            raise ValueError("CTR group_ids must be unsigned 32-bit labels matching category_hashes.")
        group_ids = np.ascontiguousarray(group_ids, dtype=np.uint32)
        ordered_groups = group_ids[permutation]
        heads = ordered_groups[np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]]
        if len(set(map(int, heads))) != len(heads):
            raise ValueError("CTR rows of each group must be contiguous in the history permutation.")
    # Stable GPU sorting preserves history order inside each category, even
    # when supplied row IDs are unrelated to the desired permutation rank.
    sorted_hashes, indices, sort_stats = stable_sort(hashes[permutation], permutation)
    boundaries = np.empty(rows, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = sorted_hashes[1:] != sorted_hashes[:-1]
    distinct = sorted_hashes[boundaries]
    categories = np.cumsum(boundaries, dtype=np.uint32) - np.uint32(1)
    values = np.empty(rows, np.float32)
    sums, counts = np.empty(len(distinct), np.float32), np.empty(len(distinct), np.uint32)
    params = _Params(rows, len(distinct), CTR_TYPES[ctr_type], int(target_border_idx), numerator, denominator)
    stats, error = _Stats(), ct.create_string_buffer(2048)
    u32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint32))
    f32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_float))
    library = _load(_build_library())
    if group_ids is None:
        code = library.cbm_compute_ctrs(ct.byref(params), u32(categories), u32(indices), f32(targets),
                                        f32(values), f32(sums), u32(counts), ct.byref(stats), error, len(error))
    else:
        code = library.cbm_compute_ctrs_grouped(ct.byref(params), u32(categories), u32(indices), f32(targets),
                                                u32(group_ids), f32(values), f32(sums), u32(counts),
                                                ct.byref(stats), error, len(error))
    if code:
        raise RuntimeError("Metal CTR computation failed: " + error.value.decode("utf-8", errors="replace"))
    return CtrResult(distinct, sums, counts, values, ctr_type, int(target_border_idx),
                     float(numerator), float(denominator),
                     {"device": stats.device_name.decode("utf-8"), "backend": "Metal",
                      "history_unit": "Group" if group_ids is not None else "Sample",
                      "kernel_dispatches": int(stats.kernel_dispatches) + sort_stats["kernel_dispatches"],
                      "gpu_seconds": stats.gpu_seconds + sort_stats["gpu_seconds"],
                      "category_sort_backend": "Metal", "category_sort": sort_stats})
