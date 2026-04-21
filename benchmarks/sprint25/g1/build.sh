#!/usr/bin/env bash
# build.sh — build Sprint 25 G1 gain-dump instrument.
#
# Produces /tmp/g1_gain_dump (arm64 Mach-O).  Links against the production
# NCatboostMlx::DispatchHistogramT2 implementation from
# catboost/mlx/methods/histogram_t2_impl.cpp so the `--kernel=t1` path matches
# bit-for-bit with the production v5 histogram path at the 18 DEC-008 configs.
#
# The Path 5 reconstruction (`--kernel=t2_path5`) is compiled into the same
# binary via headers under benchmarks/sprint25/g1/ {kernels,methods}/ and
# registers its Metal kernels under distinct names (t2_sort_s25_g1_path5,
# t2_accum_s25_g1_path5, score_splits_dump_g1) so no MLX kernel-cache
# collision occurs with production.
#
# Usage:
#   ./benchmarks/sprint25/g1/build.sh [output_path]
# Defaults to /tmp/g1_gain_dump.
#
# Exit codes:
#   0  — build succeeded
#   1  — build failed (clang++ returned non-zero)
#   2  — required path missing (MLX headers, MLX lib, or source files)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT="${1:-/tmp/g1_gain_dump}"

MLX_INC="/opt/homebrew/opt/mlx/include"
MLX_LIB="/opt/homebrew/opt/mlx/lib"

# Sanity checks ------------------------------------------------------------
if [[ ! -f "${MLX_INC}/mlx/mlx.h" ]]; then
    echo "ERROR: MLX header not found at ${MLX_INC}/mlx/mlx.h" >&2
    exit 2
fi
if [[ ! -f "${MLX_LIB}/libmlx.dylib" ]]; then
    echo "ERROR: libmlx.dylib not found at ${MLX_LIB}" >&2
    exit 2
fi

G1_DIR="${REPO_ROOT}/benchmarks/sprint25/g1"
T2_IMPL="${REPO_ROOT}/catboost/mlx/methods/histogram_t2_impl.cpp"
for f in \
    "${G1_DIR}/g1_gain_dump.cpp" \
    "${G1_DIR}/kernels/g1_kernels.h" \
    "${G1_DIR}/methods/g1_path5_dispatch.h" \
    "${T2_IMPL}" \
    "${REPO_ROOT}/catboost/mlx/kernels/kernel_sources.h"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required source missing: ${f}" >&2
        exit 2
    fi
done

# Compile ------------------------------------------------------------------
# NOTE: -I${REPO_ROOT} so `#include <catboost/mlx/kernels/kernel_sources.h>`
# resolves for BOTH g1_gain_dump.cpp and the linked histogram_t2_impl.cpp.
echo "[build] compiling g1_gain_dump -> ${OUT}"
clang++ -std=c++17 -O2 \
    -I"${REPO_ROOT}" \
    -I"${MLX_INC}" \
    -L"${MLX_LIB}" -lmlx \
    -framework Metal -framework Foundation \
    -Wno-c++20-extensions \
    "${G1_DIR}/g1_gain_dump.cpp" \
    "${T2_IMPL}" \
    -o "${OUT}"

echo "[build] done: ${OUT}"
file "${OUT}"
