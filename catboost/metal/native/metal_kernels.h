#pragma once
#include "metal_kernel_abi.h"

// Shared MSL types and CUDA-derived split selection. Objective and partition
// sources are concatenated after this string before Metal runtime compilation.
static const char* CBMMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct KernelParams {
    uint rows, features, bins, leaves;
    uint candidates, split_feature, split_bin, split_level;
    float bias, learning_rate, l2;
    uint score_function;
    uint objective, leaf_iteration, leaf_iterations, tile_rows;
    float total_weight;
    uint reserved0, reserved1, reserved2; // histogram/leaf tiles, partition tiles, score groups
    float objective_param;
    uint leaf_method, reserved3;
    float score_before_split;
};
struct SplitState {
    uint index, feature, bin, type;
    float score;
    uint valid, reserved0; // reserved0 propagates nonfinite score errors
    float gain;
};
inline void AtomicAddFloat(device atomic_uint* address, float value) {
    uint previous = atomic_load_explicit(address, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(
        address, &previous, as_type<uint>(as_type<float>(previous) + value),
        memory_order_relaxed, memory_order_relaxed)) {}
}
inline void ObjectiveReduceExpansions(threadgroup float4* high_scratch,
                                      threadgroup float4* low_scratch, uint tid);
inline void ObjectiveAddExpansion(thread float3& high, thread float3& low, float3 value);

// CUDA score_calcers.cuh widens each leaf statistic to double. L2 rounds the
// running score to float after each leaf; Cosine retains double means and
// accumulators until GetScore. Metal has no double arithmetic, so retain the
// product/division residuals in normalized float pairs. This prevents a
// left/right permutation of equal child statistics from changing split ties.
inline float2 ScorePairAdd(float2 left, float2 right) {
    const float high = left.x + right.x;
    const float part = high - left.x;
    const float error = (left.x - (high - part)) + (right.x - part);
    const float tail = error + (left.y + right.y);
    const float result = high + tail;
    const float result_part = result - high;
    return float2(result, (high - (result - result_part)) + (tail - result_part));
}
inline float ScorePairRound(float2 value) { return value.x + value.y; }
inline float2 ScorePairMultiply(float2 left, float2 right) {
    const float high = left.x * right.x;
    float low = fma(left.x, right.x, -high);
    low += left.x * right.y + left.y * right.x;
    low += left.y * right.y;
    return ScorePairAdd(float2(high, 0), float2(low, 0));
}
inline float2 ScorePairDivide(float2 numerator, float2 denominator) {
    const float high = numerator.x / denominator.x;
    float residual = fma(-high, denominator.x, numerator.x);
    residual += numerator.y - high * denominator.y;
    return ScorePairAdd(float2(high, 0), float2(residual / denominator.x, 0));
}
inline float2 ScorePairSqrt(float2 value) {
    const float high = sqrt(value.x);
    const float residual = fma(-high, high, value.x) + value.y;
    return ScorePairAdd(float2(high, 0), float2(residual / (2.0f * high), 0));
}
inline float AddL2ScoreLeaf(float score, float sum, float weight, float l2) {
    if (weight <= 1e-20f) return score;
    const float2 denominator = ScorePairAdd(float2(weight, 0), float2(l2, 0));
    const float2 mean = ScorePairDivide(float2(sum, 0), denominator);
    const float2 term = ScorePairMultiply(float2(sum, 0), mean);
    return ScorePairRound(ScorePairAdd(float2(score, 0), -term));
}
inline void AddCosineScoreLeaf(thread float2& numerator, thread float2& denominator,
                               float sum, float weight, float l2) {
    if (weight <= 0) return;
    const float2 regularized = ScorePairAdd(float2(weight, 0), float2(l2, 0));
    const float2 mean = ScorePairDivide(float2(sum, 0), regularized);
    numerator = ScorePairAdd(numerator, ScorePairMultiply(float2(sum, 0), mean));
    denominator = ScorePairAdd(denominator,
        ScorePairMultiply(ScorePairMultiply(float2(weight, 0), mean), mean));
}
inline float FinalizeCosineScore(float2 numerator, float2 denominator) {
    return -ScorePairRound(ScorePairDivide(numerator, ScorePairSqrt(denominator)));
}

// CUDA TSatL2ScoreCalcer: the rational adjustment is calculated in double,
// rounded to float, then multiplied into a double child term before the
// running float score is updated. The pole above weight=2 and its sign change
// are intentional; neither regularization nor score noise enters this score.
inline float AddSatL2ScoreLeaf(float score, float sum, float weight) {
    if (weight <= 2.0f) return score;
    float adjustment = 1.0f;
    // Above 2^25 the exact adjustment is within half a float ULP of one.
    // This also avoids overflowing weight*weight for large finite weights.
    if (weight < 33554432.0f) {
        const float2 mass = float2(weight, 0);
        const float2 numerator = ScorePairMultiply(mass, ScorePairAdd(mass, float2(-2, 0)));
        const float2 denominator = ScorePairAdd(
            ScorePairAdd(ScorePairMultiply(mass, mass), -ScorePairMultiply(mass, float2(3, 0))),
            float2(1, 0));
        adjustment = ScorePairRound(ScorePairDivide(numerator, denominator));
    }
    const float2 mean = ScorePairDivide(float2(sum, 0), float2(weight, 0));
    // Apply a large near-pole adjustment before the final product so a tiny
    // gradient square cannot underflow before it becomes a representable term.
    const float2 term = ScorePairMultiply(ScorePairMultiply(float2(sum, 0), float2(adjustment, 0)), mean);
    return ScorePairRound(ScorePairAdd(float2(score, 0), -term));
}

// Structure statistics must reduce the sampled weak target, while final leaf
// estimation recomputes the objective on the original observation weights.
kernel void ReduceStructurePartials(const device float* gradients [[buffer(0)]],
                                    const device float* weights [[buffer(1)]],
                                    const device uint* row_indices [[buffer(2)]],
                                    const device uint* offsets [[buffer(3)]],
                                    device float4* partials [[buffer(4)]],
                                    constant KernelParams& p [[buffer(5)]],
                                    uint2 local [[thread_position_in_threadgroup]],
                                    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x, leaf = group.y;
    threadgroup float4 high_scratch[256], low_scratch[256];
    float3 high = float3(0), low = float3(0);
    for (uint position = offsets[leaf] + group.x * 256 + tid; position < offsets[leaf + 1];
         position += p.reserved0 * 256) {
        const uint row = row_indices[position];
        ObjectiveAddExpansion(high, low, float3(gradients[row], 0, weights[row]));
    }
    high_scratch[tid] = float4(high, 0); low_scratch[tid] = float4(low, 0);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) {
        const uint cell = 2 * (leaf * p.reserved0 + group.x);
        partials[cell] = high_scratch[0]; partials[cell + 1] = low_scratch[0];
    }
}

// CUDA PartitionUpdateImpl's reduced parent gradient/weight statistics. The
// preceding objective kernel creates bounded partial sums; this reduction
// avoids one global atomic addition per row into a single leaf accumulator.
kernel void CollectPartitionStatistics(const device float4* partials [[buffer(0)]],
                                       device float* sums [[buffer(1)]],
                                       device float* weights [[buffer(2)]],
                                       constant KernelParams& p [[buffer(3)]],
                                       uint tid [[thread_position_in_threadgroup]],
                                       uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float4 high[256], low[256];
    const uint cell = 2 * (leaf * p.reserved0 + tid);
    high[tid] = tid < p.reserved0 ? partials[cell] : float4(0);
    low[tid] = tid < p.reserved0 ? partials[cell + 1] : float4(0);
    ObjectiveReduceExpansions(high, low, tid);
    if (tid == 0) {
        sums[leaf] = high[0].x + low[0].x;
        weights[leaf] = high[0].z + low[0].z;
    }
}

// Translation of pointwise_scores.cu::FindOptimalSplitSingleFoldImpl and
// score_calcers.cuh::{TL2ScoreCalcer,TCosineScoreCalcer}. Structure denominators
// are observation weights or Newton Hessians. Both feature types feed the
// selected histogram statistic first, then its complement, as CUDA does.
// Equality-bin scoring order is independent of the model's right-child bit.
// Float pairs preserve CUDA's wider intermediates; nonfinite scores are
// explicit errors. Per-block winner reduction preserves candidate-index ties.
kernel void FindSplitWinners(const device float* sums [[buffer(0)]],
                             const device float* weights [[buffer(1)]],
                             const device float* leaf_sums [[buffer(2)]],
                             const device float* leaf_weights [[buffer(3)]],
                             const device uint* features [[buffer(4)]],
                             const device uint* bins [[buffer(5)]],
                             const device uchar* types [[buffer(6)]],
                             device SplitState* winners [[buffer(7)]],
                             const device float* feature_noise [[buffer(8)]],
                             const device uint* feature_offsets [[buffer(9)]],
                             const device float2* feature_penalties [[buffer(10)]],
    constant KernelParams& p [[buffer(11)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    float best_score = INFINITY, best_raw_score = INFINITY;
    uint best_index = 0xffffffffu;
    uint bad = 0;
    for (uint candidate = group * 256 + tid; candidate < p.candidates;
         candidate += p.reserved2 * 256) {
        float score = 0;
        float2 numerator = 0.0f, denominator = float2(1e-10f, 0);
        for (uint leaf = 0; leaf < p.leaves; ++leaf) {
            const ulong cell = p.reserved3
                ? ulong(leaf) * feature_offsets[p.features] + feature_offsets[features[candidate]] + bins[candidate]
                : (ulong(leaf) * p.features + features[candidate]) * p.bins + bins[candidate];
            const float selected_sum = sums[cell];
            const float selected_weight = weights[cell];
            const float other_sum = leaf_sums[leaf] - selected_sum;
            const float other_weight = max(leaf_weights[leaf] - selected_weight, 0.0f);
            for (uint side = 0; side < 2; ++side) {
                const float sum = side == 0 ? selected_sum : other_sum;
                const float weight = side == 0 ? selected_weight : other_weight;
                if (p.score_function == 4) {
                    if (weight > 1e-20f) score += (-sum / weight) * sum * (1.0f + 2.0f * log(weight + 1.0f));
                } else if (p.score_function == 5) {
                    float adjust = weight > 1.0f ? weight / (weight - 1.0f) : 0.0f;
                    adjust *= adjust;
                    if (weight > 0.0f) score += -(sum * adjust) * (sum / weight);
                } else if (p.score_function == 6) {
                    score = AddSatL2ScoreLeaf(score, sum, weight);
                } else if ((p.score_function & 1u) == 0) {
                    score = AddL2ScoreLeaf(score, sum, weight, p.l2);
                } else {
                    AddCosineScoreLeaf(numerator, denominator, sum, weight, p.l2);
                }
            }
        }
        if (p.score_function == 1 || p.score_function == 3) score = FinalizeCosineScore(numerator, denominator) + feature_noise[features[candidate]];
        score *= feature_penalties[features[candidate]].x;
        const float gain = (score - p.score_before_split) * feature_penalties[features[candidate]].y;
        bad |= uint(!isfinite(score) || !isfinite(gain));
        if (isfinite(score) && isfinite(gain) && (gain < best_score || (gain == best_score && candidate < best_index))) {
            best_score = gain; best_raw_score = score; best_index = candidate;
        }
    }
    threadgroup float local_scores[256], local_raw_scores[256];
    threadgroup uint local_indices[256], local_bad[256];
    local_scores[tid] = best_score; local_raw_scores[tid] = best_raw_score; local_indices[tid] = best_index; local_bad[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint other = tid + stride;
            if (local_scores[other] < local_scores[tid] ||
                (local_scores[other] == local_scores[tid] && local_indices[other] < local_indices[tid])) {
                local_scores[tid] = local_scores[other]; local_raw_scores[tid] = local_raw_scores[other]; local_indices[tid] = local_indices[other];
            }
            local_bad[tid] |= local_bad[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const uint index = local_indices[0];
        SplitState result = {index, 0, 0, 0, local_raw_scores[0], uint(index != 0xffffffffu), local_bad[0], local_scores[0]};
        if (result.valid) { result.feature = features[index]; result.bin = bins[index]; result.type = types[index]; }
        winners[group] = result;
    }
}

// Apply a supplied shared structure one split at a time to another CTR
// permutation, preserving original target/cursor row order and leaf bit order.
kernel void UpdateFixedPermutationSplit(const device uchar* bins [[buffer(0)]],
    device uint* leaf_ids [[buffer(1)]], const device SplitState* splits [[buffer(2)]],
    constant KernelParams& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row < p.rows) {
        const SplitState split = splits[p.split_level];
        const uint bin = uint(bins[ulong(split.feature) * p.rows + row]);
        const bool right = split.type ? bin == split.bin : bin > split.bin;
        leaf_ids[row] |= uint(right) << p.split_level;
    }
}

kernel void ReduceSplitWinners(const device SplitState* partials [[buffer(0)]],
                               device SplitState* winner [[buffer(1)]],
                               constant KernelParams& p [[buffer(2)]],
                               uint tid [[thread_position_in_threadgroup]]) {
    threadgroup SplitState values[256];
    SplitState empty = {0xffffffffu, 0, 0, 0, INFINITY, 0, 0, INFINITY};
    values[tid] = tid < p.reserved2 ? partials[tid] : empty;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const SplitState other = values[tid + stride];
            const uint bad = values[tid].reserved0 | other.reserved0;
            if (other.gain < values[tid].gain ||
                (other.gain == values[tid].gain && other.index < values[tid].index)) values[tid] = other;
            values[tid].reserved0 = bad;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) winner[0] = values[0];
}
)METAL";
