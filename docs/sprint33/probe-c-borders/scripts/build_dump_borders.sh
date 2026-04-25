#!/usr/bin/env bash
# Build csv_train with -DCATBOOST_MLX_DUMP_BORDERS so it dumps quantization
# borders to stdout in `BORDER\t<feat>\t<value>` format and exits early.
#
# Usage: ./scripts/build_dump_borders.sh [output_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_probe_c}"

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
    -DCATBOOST_MLX_DUMP_BORDERS \
    -DCOSINE_T3_MEASURE \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
