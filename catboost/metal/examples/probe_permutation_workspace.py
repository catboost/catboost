"""Exercise the PR 3194 scalar permutation admission check in a fresh process.

Pass an already built standalone scalar dylib with --runtime. Cases 577k and
825k use 256 synthetic encoded columns, 128 numeric thresholds per column,
depth 10 and four permutations. This isolates workspace admission from raw
categorical preprocessing and training. --case tiny is a quick ABI check.

For example, run the accepted binary under --case 825k --expect legacy-limit
and the candidate under --case 825k --expect admitted. Redirect each JSON
report to a separate file. The equivalent --case 577k should admit both.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys

import numpy as np

METAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METAL / "python"))
from catboost_metal import _native as native


def workspace(lib, handle):
    tiles, histogram, total = ct.c_uint32(), ct.c_uint64(), ct.c_uint64()
    error = ct.create_string_buffer(1024)
    status = lib.cbm_session_get_workspace_info(
        handle, ct.byref(tiles), ct.byref(histogram), ct.byref(total), error, len(error))
    if status:
        raise RuntimeError(error.value.decode())
    return dict(tiles=tiles.value, histogram_bytes=histogram.value, estimated_bytes=total.value)


def run(runtime, case):
    lib = native._load(runtime)
    rows = {"tiny": 128, "577k": 577000, "825k": 825000}[case]
    features, borders, depth = (4, 8, 2) if case == "tiny" else (256, 128, 10)
    count = 4
    handle, error = ct.c_void_p(), ct.create_string_buffer(2048)
    name = ct.create_string_buffer(256)
    if lib.cbm_device_info(name, len(name), error, len(error)):
        raise RuntimeError(error.value.decode())
    report = dict(case=case, rows=rows, encoded_columns=features, borders=borders,
                  depth=depth, permutations=count, macos=platform.mac_ver()[0],
                  device=name.value.decode(), runtime=str(runtime),
                  runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
    # Pass one read-only host matrix four times. The runtime owns distinct GPU
    # buffers, and keeping one host input avoids measuring four Python copies.
    matrix = np.full((features, rows), 1, np.uint8)
    targets = np.zeros(rows, np.float32)
    candidates = np.repeat(np.arange(features, dtype=np.uint32), borders)
    thresholds = np.tile(np.arange(borders, dtype=np.uint32), features)
    params = native.SessionParams(
        native.TrainParams(rows, features, candidates.size, borders + 1, 1, depth, 0, .1, 3., 0.),
        0, 1, 0, 0)
    cursors = np.stack([np.full(rows, p + .25, np.float32) for p in range(count)])
    try:
        status = lib.cbm_session_create(ct.byref(params), native._u8(matrix), native._f32(targets),
            None, None, native._u32(candidates), native._u32(thresholds), None,
            ct.byref(handle), error, len(error))
        if status:
            raise RuntimeError("Create failed: " + error.value.decode())
        base = workspace(lib, handle)
        report["initial_workspace"] = base
        # Mirrors ConfigurePermutations for scalar RMSE with bootstrap off.
        additional = (count - 1) * (matrix.nbytes + 4 * rows) + 32 * depth
        report["requested_workspace_bytes"] = base["estimated_bytes"] + additional
        matrices = (ct.POINTER(ct.c_uint8) * count)(*[native._u8(matrix) for _ in range(count)])
        initial = (ct.POINTER(ct.c_float) * count)(*[native._f32(v) for v in cursors])
        status = lib.cbm_session_set_permutations(
            handle, count, matrices, initial, None, None, error, len(error))
        if status:
            message = error.value.decode()
            if message != "Permutation matrices and cursors exceed the 1 GiB GPU memory limit":
                raise RuntimeError("Unexpected permutation failure: " + message)
            report.update(outcome="legacy-limit", error=message)
        else:
            output = np.empty_like(cursors)
            lambdas, valid = np.empty(count, np.float32), np.empty(count, np.uint8)
            status = lib.cbm_session_copy_permutation_state(
                handle, count, native._f32(output), native._f32(lambdas), native._u8(valid),
                error, len(error))
            if status:
                raise RuntimeError(error.value.decode())
            np.testing.assert_array_equal(output, cursors)
            report.update(outcome="admitted", configured_workspace=workspace(lib, handle),
                          cursor_rows_verified=output.size)
    finally:
        if handle.value:
            lib.cbm_session_close(handle)
    # Darwin reports ru_maxrss in bytes.
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--case", choices=["tiny", "577k", "825k"], default="tiny")
    parser.add_argument("--expect", choices=["admitted", "legacy-limit"], default="admitted")
    args = parser.parse_args()
    result = run(args.runtime.resolve(), args.case)
    print(json.dumps(result, indent=2))
    raise SystemExit(int(result["outcome"] != args.expect))
