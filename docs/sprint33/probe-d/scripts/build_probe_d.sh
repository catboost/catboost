#!/usr/bin/env bash
# Build csv_train_probe_d for S33-PROBE-D.
#
# Flags:
#   -DCOSINE_RESIDUAL_INSTRUMENT  enables fp32-vs-fp64 double-shadow at every
#                                 (feature, bin) inside FindBestSplit, with
#                                 per-depth CSV emission via WriteCosAccumCSV.
#                                 Also bypasses the ST+Cosine guard at line 595.
#   -DPROBE_D_ARM_AT_ITER=1       arm the audit at iter=1 (1-indexed iter=2)
#                                 instead of the default iter=0. Captures
#                                 per-(feature, bin) gain at iter=2 depth 0..5.
#
# Usage: ./scripts/build_probe_d.sh [output_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_probe_d}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"
SRC="${REPO_ROOT}/catboost/mlx/tests/csv_train.cpp"

KERNEL_SOURCES="${REPO_ROOT}/catboost/mlx/kernels/kernel_sources.h"
EXPECTED_MD5="9edaef45b99b9db3e2717da93800e76f"
ACTUAL_MD5="$(md5 -q "${KERNEL_SOURCES}" 2>/dev/null || md5sum "${KERNEL_SOURCES}" | cut -d' ' -f1)"
if [[ "${ACTUAL_MD5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: kernel_sources.h md5 mismatch (got ${ACTUAL_MD5}, expected ${EXPECTED_MD5})" >&2
    exit 2
fi
echo "[build] kernel_sources.h md5 OK: ${ACTUAL_MD5}"

clang++ -std=c++17 -O2 \
    -DCOSINE_RESIDUAL_INSTRUMENT \
    -DPROBE_D_ARM_AT_ITER=1 \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
