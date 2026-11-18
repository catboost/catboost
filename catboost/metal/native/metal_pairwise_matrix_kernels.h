#pragma once

// PairLogitPairwise (objective15): edge derivatives retain ClipProb, and leaf
// projection keeps the complete Laplacian. Stable GPU sorting is supplied by
// metal_sort.h; a fixed tree's sorted leaf-pair layout is reused for trials.
static const char* CBMMetalPairwiseMatrixSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct PairwiseMatrixParams {
    uint rows, pairs, leaves, leaf_method;
    uint reserved0, reserved1, reserved2, reserved3;
};

inline void PairMatrixAdd4(thread float4& high, thread float4& low, float4 value) {
    const float4 sum = high + value;
    const float4 virtual_value = sum - high;
    const float4 error = (high - (sum - virtual_value)) + (value - virtual_value) + low;
    high = sum + error;
    low = error - (high - sum);
}
inline float PairMatrixLogOnePlus(float value) {
    const float rounded = 1.0f + value;
    return rounded == 1.0f ? value : log(rounded) * (value / (rounded - 1.0f));
}

// The caller selects sampled edge weights for structure search and original
// edge weights for the leaf oracle. Ordinary PairLogit's row bootstrap is
// intentionally not reused for this distinct target.
// Edge fields: winner gradient, Newton curvature, edge weight, positive loss.
kernel void ComputePairwiseMatrixEdges(const device float* point [[buffer(0)]],
    const device uint* winners [[buffer(1)]], const device uint* losers [[buffer(2)]],
    const device float* weights [[buffer(3)]], device float4* edges [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]], constant PairwiseMatrixParams& p [[buffer(6)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= p.pairs) return;
    const uint winner = winners[edge], loser = losers[edge];
    if (winner >= p.rows || loser >= p.rows || winner == loser) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        edges[edge] = float4(0.0f);
        return;
    }
    const float weight = weights[edge];
    if (weight == 0.0f) { edges[edge] = float4(0.0f); return; }
    const float difference = point[winner] - point[loser];
    const float exponential = exp(-abs(difference));
    const float probability = clamp(difference >= 0.0f ? 1.0f / (1.0f + exponential)
        : exponential / (1.0f + exponential), 1e-7f, 1.0f - 1e-7f);
    const float gradient = weight * (1.0f - probability);
    const float curvature = weight * probability * (1.0f - probability);
    const float loss = weight * (max(-difference, 0.0f) + PairMatrixLogOnePlus(exponential));
    const float4 result = float4(gradient, curvature, weight, loss);
    if (!all(isfinite(result)) || any(result < 0.0f) || !isfinite(difference))
        atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    edges[edge] = result;
}

// Preserve directed winner/loser cells. Same-leaf edges have zero projected
// gradient and Hessian, and CUDA's support-pair builder removes them before
// leaf estimation. Sentinel keys sort after all L*L valid cells.
kernel void BuildPairwiseLeafKeys(const device uint* leaf_ids [[buffer(0)]],
    const device uint* winners [[buffer(1)]], const device uint* losers [[buffer(2)]],
    device uint* keys [[buffer(3)]], device uint* edge_indices [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]], constant PairwiseMatrixParams& p [[buffer(6)]],
    uint edge [[thread_position_in_grid]]) {
    if (edge >= p.pairs) return;
    edge_indices[edge] = edge;
    keys[edge] = 0xffffffffu;
    if (winners[edge] >= p.rows || losers[edge] >= p.rows || winners[edge] == losers[edge]) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    const uint winner = leaf_ids[winners[edge]], loser = leaf_ids[losers[edge]];
    if (winner >= p.leaves || loser >= p.leaves) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    if (winner != loser) keys[edge] = winner * p.leaves + loser;
}

kernel void BuildPairwiseCellOffsets(const device uint* sorted_keys [[buffer(0)]],
    device uint* offsets [[buffer(1)]], constant PairwiseMatrixParams& p [[buffer(2)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell > p.leaves * p.leaves) return;
    uint first = 0, last = p.pairs;
    while (first < last) {
        const uint middle = first + (last - first) / 2;
        if (sorted_keys[middle] < cell) first = middle + 1;
        else last = middle;
    }
    offsets[cell] = first;
}

// Two consecutive float4 high/low values per directed leaf-pair cell retain
// small terms through cancellation. Dispatch L*L full256-thread groups. The
// total edge work is O(E); empty cells return before any group reduction.
kernel void ReducePairwiseLeafCells(const device uint* offsets [[buffer(0)]],
    const device uint* sorted_edge_indices [[buffer(1)]], const device float4* edges [[buffer(2)]],
    device float4* cells [[buffer(3)]], constant PairwiseMatrixParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint cell [[threadgroup_position_in_grid]]) {
    if (cell >= p.leaves * p.leaves) return;
    const uint begin = offsets[cell], end = offsets[cell + 1];
    if (begin == end) {
        if (tid == 0) { cells[2 * cell] = float4(0.0f); cells[2 * cell + 1] = float4(0.0f); }
        return;
    }
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint index = begin + tid; index < end; index += 256)
        PairMatrixAdd4(high, low, edges[sorted_edge_indices[index]]);
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) {
            high = high_parts[tid]; low = low_parts[tid];
            PairMatrixAdd4(high, low, high_parts[tid + width]);
            PairMatrixAdd4(high, low, low_parts[tid + width]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { cells[2 * cell] = high_parts[0]; cells[2 * cell + 1] = low_parts[0]; }
}

// CUDA's matrix oracle adds each directed edge to both diagonals and both
// symmetric off-diagonals. Gradient leaf estimation substitutes edge WEIGHT
// for edge curvature; it does not use an object-weight diagonal solver.
kernel void AssemblePairwiseLeafMatrix(const device float4* cells [[buffer(0)]],
    device float* gradient [[buffer(1)]], device float* hessian [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant PairwiseMatrixParams& p [[buffer(4)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= p.leaves * p.leaves) return;
    const uint row = cell / p.leaves, column = cell % p.leaves;
    const uint curvature = p.leaf_method == 1 ? 2 : 1;
    float4 high = float4(0.0f), low = float4(0.0f);
    if (row != column) {
        const uint reverse = column * p.leaves + row;
        PairMatrixAdd4(high, low, cells[2 * cell]); PairMatrixAdd4(high, low, cells[2 * cell + 1]);
        PairMatrixAdd4(high, low, cells[2 * reverse]); PairMatrixAdd4(high, low, cells[2 * reverse + 1]);
        hessian[cell] = -(high[curvature] + low[curvature]);
    } else {
        for (uint other = 0; other < p.leaves; ++other) {
            if (other == row) continue;
            const uint outgoing = row * p.leaves + other, incoming = other * p.leaves + row;
            PairMatrixAdd4(high, low, cells[2 * outgoing]); PairMatrixAdd4(high, low, cells[2 * outgoing + 1]);
            float4 incoming_high = cells[2 * incoming], incoming_low = cells[2 * incoming + 1];
            incoming_high.x = -incoming_high.x; incoming_low.x = -incoming_low.x;
            PairMatrixAdd4(high, low, incoming_high); PairMatrixAdd4(high, low, incoming_low);
        }
        gradient[row] = high.x + low.x;
        hessian[cell] = high[curvature] + low[curvature];
        if (!isfinite(gradient[row])) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    }
    if (!isfinite(hessian[cell])) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}
)METAL";
