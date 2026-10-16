"""Stable unsigned key/payload radix sorting entirely on the Metal GPU."""

import ctypes as ct
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import subprocess

import numpy as np


class _Stats(ct.Structure):
    _fields_ = [("kernel_dispatches", ct.c_uint64), ("gpu_seconds", ct.c_double),
                ("device_name", ct.c_char * 256)]


def _build_library():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise RuntimeError("Metal radix sort requires macOS on Apple Silicon.")
    import fcntl

    root = Path(__file__).resolve().parents[2]
    native = root / "native"
    digest = hashlib.sha256(platform.platform().encode())
    for name in ("metal_sort.h", "metal_sort.mm", "metal_sort_kernels.h"):
        digest.update((native / name).read_bytes())
    build = root / ".build"
    build.mkdir(exist_ok=True)
    destination = build / f"libcatboost_metal_sort_{digest.hexdigest()[:20]}.dylib"
    with (build / "sort-build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.dylib")
            try:
                process = subprocess.run([
                    "xcrun", "clang++", "-std=c++17", "-O2", "-fobjc-arc", "-dynamiclib",
                    "-framework", "Foundation", "-framework", "Metal",
                    str(native / "metal_sort.mm"), "-o", str(temporary),
                ], capture_output=True, text=True, check=False)
                if process.returncode:
                    raise RuntimeError("Could not build Metal radix sort:\n" + process.stderr)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    return destination


@lru_cache(maxsize=4)
def _load(path):
    library = ct.CDLL(str(path))
    u32 = ct.POINTER(ct.c_uint32)
    library.cbm_sort_u32.argtypes = [u32, u32, ct.c_uint32, u32, u32,
                                    ct.POINTER(_Stats), ct.c_char_p, ct.c_size_t]
    library.cbm_sort_u32.restype = ct.c_int
    return library


def _uint32_vector(values, name):
    values = np.asarray(values)
    if (values.ndim != 1 or (values.size and values.dtype.kind not in "ui") or
            (values.size and (np.any(values < 0) or np.any(values > np.iinfo(np.uint32).max)))):
        raise ValueError(f"{name} must be a one-dimensional vector of unsigned 32-bit integers.")
    if len(values) > (1 << 24):
        raise ValueError("Metal radix sort supports at most 16777216 rows.")
    return np.ascontiguousarray(values, dtype=np.uint32)


def stable_sort(keys, payload=None):
    """Return ``(sorted_keys, sorted_payload, stats)`` without modifying inputs.

    Keys and payloads must fit uint32. If payload is omitted the GPU generates
    original row IDs, so the result is a stable argsort as well as sorted keys.
    Equal keys retain input order, regardless of the payload values. Validation
    and copies happen on the host; radix ranking, histogram scans, and all
    permutations run on Metal. There is no CPU sorting fallback.
    """
    keys = _uint32_vector(keys, "keys")
    if payload is not None:
        payload = _uint32_vector(payload, "payload")
        if payload.shape != keys.shape:
            raise ValueError("payload must have the same shape as keys.")
    sorted_keys, sorted_payload = np.empty_like(keys), np.empty_like(keys)
    stats, error = _Stats(), ct.create_string_buffer(2048)
    u32 = lambda array: array.ctypes.data_as(ct.POINTER(ct.c_uint32))
    library = _load(_build_library())
    code = library.cbm_sort_u32(u32(keys), None if payload is None else u32(payload), len(keys),
                                 u32(sorted_keys), u32(sorted_payload), ct.byref(stats), error, len(error))
    if code:
        raise RuntimeError("Metal radix sort failed: " + error.value.decode("utf-8", errors="replace"))
    return sorted_keys, sorted_payload, {
        "backend": "Metal", "device": stats.device_name.decode("utf-8"),
        "kernel_dispatches": int(stats.kernel_dispatches), "gpu_seconds": stats.gpu_seconds,
        "radix_bits": 4, "radix_passes": 8 if len(keys) else 0,
    }
