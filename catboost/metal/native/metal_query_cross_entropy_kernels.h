#pragma once

// CUDA: targets/kernel/query_cross_entropy.cu. QueryCrossEntropy needs a
// diagonal point term PLUS a query Laplacian; it is not a rowwise loss.
static const char* CBMMetalQueryCrossEntropySource = R"METAL(
#include <metal_stdlib>
using namespace metal;
#ifndef CBM_QCE_THREADS
#define CBM_QCE_THREADS 256
#endif

struct QCEParams { uint rows, groups, leaves; float alpha; };

inline float QCEProbability(float x) {
    const float e = exp(x);
    return clamp(isfinite(1.0f + e) ? e / (1.0f + e) : 1.0f, 1e-7f, 1.0f - 1e-7f);
}

inline float QCELogLoss(float target, float point) {
    const float tail = exp(-abs(point));
    const float total = 1.0f + tail;
    // Metal has no log1p; retain the rounding remainder when 1+tail rounds to 1.
    const float log_tail = log(total) + (tail - (total - 1.0f)) / total;
    return max(point, 0.0f) - target * point + log_tail;
}

inline void QCEReduce(threadgroup float4* scratch, uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = CBM_QCE_THREADS / 2; stride; stride >>= 1) {
        if (lane < stride) scratch[lane] += scratch[lane + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// row_stats = (negative gradient, point Hessian, shifted Hessian, weighted loss)
// group_stats = (shift, sum shifted Hessians, weighted loss, sum weights).
// Scale is already selected by (query size, count of target > .5).
kernel void QueryCrossEntropyStatistics(
    const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device float* point [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    const device float* scales [[buffer(4)]],
    device float4* row_stats [[buffer(5)]],
    device float4* group_stats [[buffer(6)]],
    device uint* single_class [[buffer(7)]],
    constant QCEParams& p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
    threadgroup float4 scratch[CBM_QCE_THREADS];
    const uint begin = offsets[group], count = offsets[group + 1] - begin;
    const bool active = lane < count;
    const uint row = begin + min(lane, count - 1);
    const float y = targets[row], w = active ? weights[row] : 0.0f;
    const float x = point[row], scale = scales[group];
    scratch[lane] = float4(active && abs(y - targets[begin]) > 1e-5f ? 1.0f : 0.0f, 0, 0, 0);
    QCEReduce(scratch, lane);
    const bool single = scratch[0].x == 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float left = -20.0f, right = 20.0f, shift = 0.0f;
    if (!single) {
        for (uint iteration = 0; iteration < 13; ++iteration) {
            const float probability = QCEProbability(x * scale + shift);
            scratch[lane] = float4(w * (y - probability), w * (1.0f - probability) * probability, 0, 0);
            QCEReduce(scratch, lane);
            const float gradient = scratch[0].x, hessian = scratch[0].y;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (gradient > 0.0f) left = shift; else right = shift;
            if (iteration < 8) {
                shift = (left + right) * 0.5f;
            } else {
                shift += gradient / (hessian + 1e-9f);
                if (shift > right) shift = .1f * left + .9f * right;
                if (shift < left) shift = .9f * left + .1f * right;
            }
        }
    }
    const float shifted = x * scale + shift;
    const float probability = QCEProbability(x), shifted_probability = QCEProbability(shifted);
    const float gradient = w * ((1.0f - p.alpha) * (y - probability)
        + (single ? 0.0f : p.alpha * (y - shifted_probability) * scale));
    const float point_hessian = w * (1.0f - p.alpha) * probability * (1.0f - probability);
    const float shifted_hessian = single ? 0.0f
        : w * p.alpha * shifted_probability * (1.0f - shifted_probability) * scale * scale;
    const float loss = w == 0.0f ? 0.0f : w * ((1.0f - p.alpha) * QCELogLoss(y, x)
        + (single ? 0.0f : p.alpha * QCELogLoss(y, shifted)));
    if (active) row_stats[row] = float4(gradient, point_hessian, shifted_hessian, loss);
    scratch[lane] = float4(shifted_hessian, loss, w, 0);
    QCEReduce(scratch, lane);
    if (lane == 0) {
        group_stats[group] = float4(shift, scratch[0].x, scratch[0].y, scratch[0].z);
        single_class[group] = single;
    }
}

// Project the exact CUDA pair matrix into tree leaves without materializing
// n*(n-1)/2 edges per query. For A_l = sum shifted curvature in leaf l,
// the query Laplacian contributes A_l*(S-A_l)/(S+eps) on its diagonal and
// -A_l*A_m/(S+eps) off diagonal. This diagnostic implementation favors an
// explicit contract; production training can tile/reuse the per-query sums.
kernel void ProjectQueryCrossEntropyLeaves(
    const device float4* row_stats [[buffer(0)]],
    const device float4* group_stats [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device uint* leaf_ids [[buffer(3)]],
    device float* gradient [[buffer(4)]],
    device float* hessian [[buffer(5)]],
    constant QCEParams& p [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.leaves * p.leaves) return;
    const uint a = index / p.leaves, b = index % p.leaves;
    float total = 0.0f, g = 0.0f;
    for (uint group = 0; group < p.groups; ++group) {
        float ac = 0.0f, bc = 0.0f, complement = 0.0f, point_diag = 0.0f;
        for (uint row = offsets[group]; row < offsets[group + 1]; ++row) {
            const uint leaf = leaf_ids[row];
            const float4 s = row_stats[row];
            if (leaf == a) { ac += s.z; if (a == b) { g += s.x; point_diag += s.y; } }
            else complement += s.z;
            if (leaf == b) bc += s.z;
        }
        const float sum = group_stats[group].y;
        total += a == b ? point_diag : 0.0f;
        if (sum > 1e-20f) total += (a == b ? ac * complement : -ac * bc) / (sum + 1e-20f);
    }
    if (a == b) gradient[a] = g;
    hessian[index] = total;
}

// Production projection caches one four-statistic tuple per (query, leaf),
// avoiding another observation scan for every pair of leaves. Cache groups can
// be tiled by the caller when Q*L would exceed its workspace budget.
kernel void CacheQueryCrossEntropyLeafSums(
    const device float4* row_stats [[buffer(0)]],
    const device uint* offsets [[buffer(1)]],
    const device uint* leaf_ids [[buffer(2)]],
    device float4* query_leaf_sums [[buffer(3)]],
    constant QCEParams& p [[buffer(4)]],
    uint2 job [[threadgroup_position_in_grid]],
    uint2 local [[thread_position_in_threadgroup]]) {
    threadgroup float4 scratch[CBM_QCE_THREADS];
    const uint lane = local.x;
    const uint leaf = job.x, group = job.y, row = offsets[group] + lane;
    float4 value = float4(0.0f);
    if (row < offsets[group + 1]) {
        const float4 stats = row_stats[row];
        value = leaf_ids[row] == leaf ? float4(stats.xyz, 0.0f) : float4(0, 0, 0, stats.z);
    }
    scratch[lane] = value;
    QCEReduce(scratch, lane);
    if (lane == 0) query_leaf_sums[ulong(group) * p.leaves + leaf] = scratch[0];
}

kernel void SumQueryCrossEntropyLeafMatrix(
    const device float4* query_leaf_sums [[buffer(0)]],
    const device float4* group_stats [[buffer(1)]],
    device float* gradient [[buffer(2)]],
    device float* hessian [[buffer(3)]],
    constant QCEParams& p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.leaves * p.leaves) return;
    const uint a = index / p.leaves, b = index % p.leaves;
    float h = 0.0f, hlow = 0.0f, g = 0.0f, glow = 0.0f;
    for (uint group = 0; group < p.groups; ++group) {
        const float4 left = query_leaf_sums[ulong(group) * p.leaves + a];
        const float4 right = query_leaf_sums[ulong(group) * p.leaves + b];
        const float total = group_stats[group].y;
        float value = a == b ? left.y : 0.0f;
        if (total > 1e-20f) value += left.z * (a == b ? left.w : -right.z) / (total + 1e-20f);
        // Kahan across queries protects small diagonal contributions against
        // a large accumulated curvature, without device float64 support.
        const float next_h = h + (value - hlow);
        hlow = (next_h - h) - (value - hlow);
        h = next_h;
        if (a == b) {
            const float next_g = g + (left.x - glow);
            glow = (next_g - g) - (left.x - glow);
            g = next_g;
        }
    }
    if (a == b) gradient[a] = g;
    hessian[index] = h;
}
)METAL";
