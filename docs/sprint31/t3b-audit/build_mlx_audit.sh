#!/usr/bin/env bash
# build_mlx_audit.sh — build the S31-T3b ITER1_AUDIT binary.
#
# Compiles csv_train.cpp with -DITER1_AUDIT to enable per-layer iter=1 dump:
#   - Parent partition aggregates (sumG, sumH, W, leafCount) at each depth
#   - Top-K=5 split candidates (feat_idx, bin_idx, gain) per depth
#   - Winning split tuple per depth
# Output: written to JSON at docs/sprint31/t3b-audit/data/mlx_splits_seed<N>.json
#
# Also adds -DCOSINE_T3_MEASURE to bypass the ST+Cosine measurement guard.
#
# Usage:
#   ./docs/sprint31/t3b-audit/build_mlx_audit.sh [output_path]
# Default output: ./csv_train_t3b (at repo root)
#
# Exit codes:
#   0 — build succeeded
#   1 — clang++ returned non-zero
#   2 — required path missing (MLX headers, MLX lib, or source file)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-${REPO_ROOT}/csv_train_t3b}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"
SRC="${REPO_ROOT}/catboost/mlx/tests/csv_train.cpp"

# Sanity checks ---------------------------------------------------------------
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

# Compile ---------------------------------------------------------------------
echo "[build] compiling csv_train_t3b -> ${OUT}"
clang++ -std=c++17 -O2 \
    -DITER1_AUDIT \
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
