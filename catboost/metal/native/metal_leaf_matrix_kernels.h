#pragma once

// Coupled leaf oracle arithmetic from matrix_per_tree_oracle_base.h and
// descent_helpers.cpp. CUDA forms this system in host double precision; Metal
// uses compensated float pairs so a small ridge is retained beside large H.
// This is LEAF regularization, not pairwise split-score stabilization.
static const char* CBMMetalLeafMatrixSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct LeafMatrixParams {
    uint leaves, has_diagonal_part, reserved0, reserved1;
    float l2, non_diag_l2, min_leaf_weight, step;
};

inline float2 LeafMatrixNormalize(float high, float low) {
    const float value = high + low;
    const float virtual_low = value - high;
    return float2(value, (high - (value - virtual_low)) + (low - virtual_low));
}
inline float2 LeafMatrixAdd(float2 a, float2 b) {
    const float high = a.x + b.x;
    const float virtual_b = high - a.x;
    const float low = (a.x - (high - virtual_b)) + (b.x - virtual_b) + a.y + b.y;
    return LeafMatrixNormalize(high, low);
}
inline float2 LeafMatrixMultiply(float2 a, float2 b) {
    const float high = a.x * b.x;
    const float low = fma(a.x, b.x, -high) + a.x * b.y + a.y * b.x + a.y * b.y;
    return LeafMatrixNormalize(high, low);
}
inline float2 LeafMatrixDivide(float2 a, float2 b) {
    const float first = a.x / b.x;
    const float2 remainder = LeafMatrixAdd(a, -LeafMatrixMultiply(b, float2(first, 0.0f)));
    const float second = (remainder.x + remainder.y) / b.x;
    const float2 result = LeafMatrixNormalize(first, second);
    const float2 residual = LeafMatrixAdd(a, -LeafMatrixMultiply(b, result));
    return LeafMatrixAdd(result, float2((residual.x + residual.y) / b.x, 0.0f));
}
inline float2 LeafMatrixSqrt(float2 a) {
    const float root = sqrt(a.x);
    const float2 residual = LeafMatrixAdd(a, -LeafMatrixMultiply(float2(root, 0.0f), float2(root, 0.0f)));
    return LeafMatrixNormalize(root, (residual.x + residual.y) / (2.0f * root));
}
inline uint LeafMatrixDimension(constant LeafMatrixParams& p) {
    return p.leaves - uint(!p.has_diagonal_part);
}

// Input is a COMPLETE projected Hessian, including both Laplacian diagonal
// and off-diagonal cells, plus any QCE point diagonal. The output's physical
// row stride stays leaves even when the last coordinate is removed.
kernel void RegularizeLeafMatrix(const device float* hessian [[buffer(0)]],
    device float2* workspace [[buffer(1)]], constant LeafMatrixParams& p [[buffer(2)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= p.leaves * p.leaves) return;
    const uint row = cell / p.leaves, column = cell % p.leaves;
    const uint n = LeafMatrixDimension(p);
    if (row >= n || column >= n) { workspace[cell] = float2(0.0f); return; }
    const float2 prior = LeafMatrixDivide(float2(p.non_diag_l2, 0.0f), float2(float(p.leaves), 0.0f));
    float2 value = float2(hessian[cell], 0.0f);
    if (row == column) {
        if (hessian[cell] == 0.0f) value = float2(10.0f, 0.0f);
        value = LeafMatrixAdd(value, float2(p.l2, 0.0f));
        value = LeafMatrixAdd(value, LeafMatrixAdd(float2(p.non_diag_l2, 0.0f), -prior));
    } else value = LeafMatrixAdd(value, -prior);
    workspace[cell] = value;
}

// One full256-thread group, with a device-backed lower triangle. Columns are
// sequential; independent rows within a column run together. All arithmetic
// uses high/low pairs, including factorization and forward/back substitution.
// Status bits:1 nonfinite/asymmetric input;2 nonpositive pivot;4 nonfinite
// output. Status must be cleared by the command owner before dispatch.
kernel void SolveLeafMatrix(device float2* matrix [[buffer(0)]],
    const device float* gradient [[buffer(1)]], device float* direction [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant LeafMatrixParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint matrix_id [[threadgroup_position_in_grid]]) {
    matrix += matrix_id * p.leaves * p.leaves;
    gradient += matrix_id * p.leaves;
    direction += matrix_id * p.leaves;
    status += matrix_id;
    const uint n = LeafMatrixDimension(p), stride = p.leaves;
    threadgroup float2 solution[256];
    threadgroup float2 pivot;
    if (tid < stride) direction[tid] = 0.0f;
    if (tid < n) {
        solution[tid] = float2(gradient[tid], 0.0f);
        if (!isfinite(gradient[tid])) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        for (uint column = 0; column < n; ++column) {
            const float2 cell = matrix[tid * stride + column];
            const float2 mirror = matrix[column * stride + tid];
            if (!all(isfinite(cell)) || any(cell != mirror))
                atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (atomic_load_explicit(status, memory_order_relaxed)) return;
    for (uint column = 0; column < n; ++column) {
        if (tid == 0) {
            float2 diagonal = matrix[column * stride + column];
            for (uint k = 0; k < column; ++k)
                diagonal = LeafMatrixAdd(diagonal, -LeafMatrixMultiply(matrix[column * stride + k], matrix[column * stride + k]));
            if (!all(isfinite(diagonal))) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
            if (!(diagonal.x > 0.0f || (diagonal.x == 0.0f && diagonal.y > 0.0f)))
                atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
            pivot = LeafMatrixSqrt(diagonal);
            matrix[column * stride + column] = pivot;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (atomic_load_explicit(status, memory_order_relaxed)) return;
        if (tid > column && tid < n) {
            float2 cell = matrix[tid * stride + column];
            for (uint k = 0; k < column; ++k)
                cell = LeafMatrixAdd(cell, -LeafMatrixMultiply(matrix[tid * stride + k], matrix[column * stride + k]));
            matrix[tid * stride + column] = LeafMatrixDivide(cell, pivot);
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    // Forward substitution: after solving row i, all later right-hand sides
    // receive its contribution in parallel, preserving a fixed update order.
    for (uint row = 0; row < n; ++row) {
        if (tid == row) solution[row] = LeafMatrixDivide(solution[row], matrix[row * stride + row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid > row && tid < n)
            solution[tid] = LeafMatrixAdd(solution[tid], -LeafMatrixMultiply(matrix[tid * stride + row], solution[row]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint reverse = 0; reverse < n; ++reverse) {
        const uint row = n - 1 - reverse;
        if (tid == row) solution[row] = LeafMatrixDivide(solution[row], matrix[row * stride + row]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < row)
            solution[tid] = LeafMatrixAdd(solution[tid], -LeafMatrixMultiply(matrix[row * stride + tid], solution[row]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < n) {
        const float value = solution[tid].x + solution[tid].y;
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
        direction[tid] = value;
    }
}

// RegularizeImpl masks the UPDATED point, not the coupled Newton direction.
// Keep the fixed last coordinate at zero until final NeedZeroAverage centering.
kernel void UpdateLeafMatrixPoint(const device float* point [[buffer(0)]],
    const device float* direction [[buffer(1)]], const device float* leaf_weights [[buffer(2)]],
    device float* updated [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant LeafMatrixParams& p [[buffer(5)]], uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    if (!isfinite(leaf_weights[leaf]) || leaf_weights[leaf] < 0.0f)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    if (leaf >= LeafMatrixDimension(p) || leaf_weights[leaf] < p.min_leaf_weight) {
        updated[leaf] = 0.0f;
        return;
    }
    const float2 candidate = LeafMatrixAdd(float2(point[leaf], 0.0f),
        LeafMatrixMultiply(float2(p.step, 0.0f), float2(direction[leaf], 0.0f)));
    const float value = candidate.x + candidate.y;
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    updated[leaf] = value;
}

kernel void ReduceLeafMatrixDirectionalDot(const device float* gradient [[buffer(0)]],
    const device float* direction [[buffer(1)]], device float2* result [[buffer(2)]],
    constant LeafMatrixParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float2 partials[256];
    partials[tid] = tid < LeafMatrixDimension(p)
        ? LeafMatrixMultiply(float2(gradient[tid], 0.0f), float2(direction[tid], 0.0f)) : float2(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset; offset >>= 1) {
        if (tid < offset) partials[tid] = LeafMatrixAdd(partials[tid], partials[tid + offset]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[0] = partials[0];
}
)METAL";
