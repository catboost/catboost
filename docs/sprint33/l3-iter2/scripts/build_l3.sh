#!/usr/bin/env bash
# build_l3.sh — build the S33-L3-ITER2 instrumented binary.
#
# Compiles csv_train.cpp with -DL3_ITER2_DUMP to enable env-var-guarded
# per-stage binary dumps at iter=1 (0-indexed) = 1-indexed iter=2.
#
# Env-var dump gates (runtime, not compile-time):
#   CATBOOST_MLX_DUMP_ITER2_GRAD=<dir>    S1 gradient/hessian binary
#   CATBOOST_MLX_DUMP_ITER2_HIST=<dir>    S2 histogram + partition stats binary
#   CATBOOST_MLX_DUMP_ITER2_TREE=<dir>    S2 best-split JSON
#   CATBOOST_MLX_DUMP_ITER2_LEAVES=<dir>  S3 leaf values JSON + partitions binary
#   CATBOOST_MLX_DUMP_ITER2_APPROX=<dir>  S4 cursor binary
#
# Usage:
#   ./docs/sprint33/l3-iter2/scripts/build_l3.sh [output_path]
# Default output: ./csv_train_l3 (at repo root)
#
# Exit codes:
#   0 — build succeeded
#   1 — clang++ returned non-zero
#   2 — required path missing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_l3}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"
SRC="${REPO_ROOT}/catboost/mlx/tests/csv_train.cpp"

if [[ ! -f "${MLX_INC}/mlx/mlx.h" ]]; then
    echo "ERROR: MLX header not found at ${MLX_INC}/mlx/mlx.h" >&2
    exit 2
fi
if [[ ! -f "${MLX_LIB}/libmlx.dylib" ]]; then
    echo "ERROR: libmlx.dylib not found at ${MLX_LIB}" >&2
    exit 2
fi
if [[ ! -f "${SRC}" ]]; then
    echo "ERROR: source not found: ${SRC}" >&2
    exit 2
fi

# Verify v5 kernel md5 is unchanged before building.
KERNEL_SOURCES="${REPO_ROOT}/catboost/mlx/kernels/kernel_sources.h"
EXPECTED_MD5="9edaef45b99b9db3e2717da93800e76f"
ACTUAL_MD5="$(md5 -q "${KERNEL_SOURCES}" 2>/dev/null || md5sum "${KERNEL_SOURCES}" | cut -d' ' -f1)"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: kernel_sources.h md5 mismatch!" >&2
    echo "  expected: ${EXPECTED_MD5}" >&2
    echo "  actual:   ${ACTUAL_MD5}" >&2
    echo "  Kernel sources have been modified — aborting build." >&2
    exit 2
fi
echo "[build] kernel_sources.h md5 OK: ${ACTUAL_MD5}"

echo "[build] compiling csv_train_l3 -> ${OUT}"
clang++ -std=c++17 -O2 \
    -DL3_ITER2_DUMP \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
echo "[build] kernel_sources.h md5 (post-build): $(md5 -q "${KERNEL_SOURCES}" 2>/dev/null || md5sum "${KERNEL_SOURCES}" | cut -d' ' -f1)"
