#!/usr/bin/env bash
# build_mlx_terms.sh — build the S32-T2 COSINE_TERM_AUDIT binary.
#
# Compiles csv_train.cpp with -DCOSINE_TERM_AUDIT to enable per-bin term dump
# at depth=0 iter=0 for SymmetricTree+Cosine:
#   (feat, bin, sumLeft, sumRight, weightLeft, weightRight, lambda,
#    cosNum_term, cosDen_term, gain)
#
# Also adds -DCOSINE_T3_MEASURE to bypass the ST+Cosine measurement guard.
#
# Usage:
#   ./docs/sprint32/t2-terms/build_mlx_terms.sh [output_path]
# Default output: ./csv_train_t2_terms (at repo root)
#
# Exit codes:
#   0 — build succeeded
#   1 — clang++ returned non-zero
#   2 — required path missing

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_t2_terms}"

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

echo "[build] compiling csv_train_t2_terms -> ${OUT}"
clang++ -std=c++17 -O2 \
    -DCOSINE_TERM_AUDIT \
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
