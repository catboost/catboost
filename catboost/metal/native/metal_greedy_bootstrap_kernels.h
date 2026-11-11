#pragma once
#include <stdint.h>

struct CBMGreedyBootstrapParams {
    uint32_t Rows, Leaves, BootstrapType, Reserved;
};
static_assert(sizeof(CBMGreedyBootstrapParams) == 16);

// CUDA BootstrapAndFilter physically removes Bernoulli/Poisson zero draws before
// greedy structure search. Retain all-row partitions for histogram/routing work,
// but supply these virtual sampled offsets ONLY to SelectGreedyLeaves so its
// min_data_in_leaf terminal check counts the same sampled observations.
//
// Type: 0=No, 1=Bayesian, 2=Bernoulli, 3=Poisson. No/Bayesian retain every row.
// Bernoulli/Poisson retain positive multipliers, including original-zero-weight
// observations. No original sample-weight buffer is involved. CUDA's filter uses
// abs(draw)>1e-20; positive Bernoulli/Poisson draws are integers, making this
// equivalent to draw>0 for those supported distributions.
//
// Host contract: 0<rows<=2^24, 0<leaves<=65536, type<=3, reserved=0; row_indices
// is a permutation of [0,rows); offsets[leaves+1] is nondecreasing from 0 to rows;
// multipliers[rows] is finite and nonnegative. Input/output offsets are distinct.
// Empty leaves and an entirely excluded draw are valid; no sampling retry occurs.
//
// Dispatch Count with leaves groups of 256 threads, then Prefix with one group
// of 256 threads in a separate ordered encoder. Output sampled_offsets[leaves+1]
// contains a prefix sum of retained counts, and sampled_offsets[leaves] is their
// total. No additional device scratch is required. Both passes use 1 KiB local.
static const char* CBMMetalGreedyBootstrapSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GreedyBootstrapParams { uint rows, leaves, bootstrap_type, reserved; };

kernel void CountGreedyBootstrapRows(
    const device float* multipliers [[buffer(0)]],
    const device uint* row_indices [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    device uint* sampled_offsets [[buffer(3)]],
    constant GreedyBootstrapParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    if (p.bootstrap_type < 2) {
        if (tid == 0) {
            sampled_offsets[leaf + 1] = offsets[leaf + 1] - offsets[leaf];
            if (leaf == 0) sampled_offsets[0] = 0;
        }
        return;
    }
    threadgroup uint counts[256];
    uint count = 0;
    for (uint index = offsets[leaf] + tid; index < offsets[leaf + 1]; index += 256)
        count += uint(multipliers[row_indices[index]] > 0.0f);
    counts[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) counts[tid] += counts[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        sampled_offsets[leaf + 1] = counts[0];
        if (leaf == 0) sampled_offsets[0] = 0;
    }
}

kernel void PrefixGreedyBootstrapOffsets(
    device uint* sampled_offsets [[buffer(0)]],
    constant GreedyBootstrapParams& p [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint prefix[256];
    threadgroup uint chunk_total;
    uint carry = 0;
    for (uint begin = 0; begin < p.leaves; begin += 256) {
        const uint leaf = begin + tid;
        prefix[tid] = leaf < p.leaves ? sampled_offsets[leaf + 1] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 256; stride <<= 1) {
            const uint add = tid >= stride ? prefix[tid - stride] : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[tid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (leaf < p.leaves) sampled_offsets[leaf + 1] = carry + prefix[tid];
        if (tid == 255) chunk_total = prefix[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        carry += chunk_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Scalar CUDA greedy noise uses already bootstrapped gradients and structure
// weights, retaining only w>1e-15. Its variance is sum(g*g/w)/sum(w), not the
// symmetric trainer's pre-bootstrap row-count normalization. Dispatch any
// positive number of groups of 256 threads. float4 output per group is
// (numerator_high, weight_high, numerator_low, weight_low); sum both parts in
// host double precision. No first moment is needed by the CUDA score helper.
inline void GreedyNoiseAdd(thread float2& high, thread float2& low, float2 value) {
    const float2 sum = high + value;
    const float2 value_part = sum - high;
    const float2 error = (high - (sum - value_part)) + (value - value_part);
    const float2 tail = low + error;
    const float2 next = sum + tail;
    const float2 tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

kernel void ReduceGreedyScoreNoiseStatistics(
    const device float* gradient [[buffer(0)]],
    const device float* structure_weight [[buffer(1)]],
    device float4* partials [[buffer(2)]],
    constant GreedyBootstrapParams& p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 scratch[256];
    float2 high = 0.0f, low = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float w = structure_weight[row];
        if (w > 1e-15f) {
            const float g = gradient[row];
            GreedyNoiseAdd(high, low, float2(g * g / w, w));
        }
    }
    scratch[tid] = float4(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = scratch[tid].xy; low = scratch[tid].zw;
            GreedyNoiseAdd(high, low, scratch[tid + stride].xy);
            GreedyNoiseAdd(high, low, scratch[tid + stride].zw);
            scratch[tid] = float4(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = scratch[0];
}
)METAL";
