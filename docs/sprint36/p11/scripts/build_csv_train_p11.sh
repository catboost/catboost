#!/usr/bin/env bash
# S36-LATENT-P11 — build csv_train_p11 from current master csv_train.cpp.
# This is a vanilla build (no instrumentation flags); the binary should be
# bit-equivalent in behaviour to a fresh `csv_train` rebuilt from current source.
#
# Output: catboost-mlx/csv_train_p11
# Kernel md5 invariant: 9edaef45b99b9db3e2717da93800e76f (must match — we are
# NOT applying the P11 fix in this probe, only measuring drift on the unfixed
# baseline).
#
# Usage: ./docs/sprint36/p11/scripts/build_csv_train_p11.sh [output_path]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_p11}"

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
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${SRC}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
