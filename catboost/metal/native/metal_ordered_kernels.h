#pragma once

// Numeric, single-device Ordered foundation. See ../ORDERED_PORT.md for the
// CUDA source map and the persistent-session contract. This does not enable
// Ordered in the public trainer by itself.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

struct CBMOrderedFold {
    uint32_t EstimateEnd;
    uint32_t QualityEnd;
    uint32_t CursorOffset;
    uint32_t Reserved = 0;
};
static_assert(sizeof(CBMOrderedFold) == 16);

inline std::vector<CBMOrderedFold> CBMCreateNumericOrderedFolds(
    uint32_t rows, float growthRate = 2.0f, uint32_t minFoldSize = 100) {
    if (rows < 4 || rows > (1u << 24) || !minFoldSize ||
        !std::isfinite(growthRate) || growthRate <= 1.0f) {
        throw std::invalid_argument("Invalid numeric Ordered fold configuration");
    }
    uint32_t prefix = 1;
    if (rows >= 500) {
        const uint32_t ratio = (uint64_t(rows) + minFoldSize - 1) / minFoldSize;
        const uint32_t folds = static_cast<uint32_t>(std::ceil(std::log2(ratio)));
        prefix = folds >= 18 ? (rows + (1u << 18) - 1) / (1u << 18)
                             : std::min(minFoldSize, rows / 50);
    }
    // TTrivialQueriesGrouping::NextQueryOffsetForLine chooses the NEXT row,
    // even though fold ranges themselves use exclusive right endpoints.
    prefix = std::min(prefix + 1, rows);
    uint64_t offset = 0;
    std::vector<CBMOrderedFold> result;
    while (prefix < rows) {
        const uint32_t truncated = static_cast<uint32_t>(
            std::min(double(rows), double(prefix) * growthRate));
        const uint32_t end = std::min(truncated + 1, rows);
        if (result.size() >= 4096 || offset + end > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("Ordered fold workspace exceeds foundation limits");
        }
        result.push_back({prefix, end, static_cast<uint32_t>(offset), 0});
        offset += end;
        prefix = end;
    }
    return result;
}

// CUDA samples_grouping.h::NextQueryOffsetForLine returns the end of the
// containing group, including the next group when line is already a boundary.
// Ends are in this permutation's group order, not the original Pool order.
inline std::vector<CBMOrderedFold> CBMCreateGroupedOrderedFolds(
    uint32_t rows, const std::vector<uint32_t>& ends, double growthRate, uint32_t minFoldSize) {
    if (rows < 4 || rows > (1u << 24) || ends.size() < 4 || ends.back() != rows || !minFoldSize ||
        !std::isfinite(growthRate) || growthRate <= 1.0f)
        throw std::invalid_argument("Ordered requires at least four groups and valid fold options");
    uint32_t previous = 0;
    for (uint32_t end : ends) {
        if (end <= previous || end > rows) throw std::invalid_argument("Ordered group ends must partition all rows");
        previous = end;
    }
    const auto nextEnd = [&](uint32_t line) {
        const auto it = std::upper_bound(ends.begin(), ends.end(), line);
        return it == ends.end() ? rows : *it;
    };
    uint32_t prefix = 1;
    if (rows >= 500) {
        const uint32_t ratio = (uint64_t(rows) + minFoldSize - 1) / minFoldSize;
        const uint32_t folds = static_cast<uint32_t>(std::ceil(std::log2(ratio)));
        prefix = folds >= 18 ? (rows + (1u << 18) - 1) / (1u << 18) : std::min(minFoldSize, rows / 50);
    }
    prefix = nextEnd(prefix);
    uint64_t offset = 0;
    std::vector<CBMOrderedFold> result;
    do {
        const uint32_t end = nextEnd(static_cast<uint32_t>(std::min(double(rows), double(prefix) * growthRate)));
        if (result.size() >= 4096 || offset + end > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("Ordered fold workspace exceeds foundation limits");
        result.push_back({prefix, end, static_cast<uint32_t>(offset), 0});
        offset += end; prefix = end;
    } while (prefix < rows);
    return result;
}

static const char* CBMMetalOrderedSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct OrderedParams {
    uint rows, features, folds, leaves;
    uint candidates, packed_rows, test_only, score_function;
    float l2;
    uint normalize;
    float score_before, learning_rate;
};

// Fold uint4 = estimate_end, quality_end, cursor_offset, reserved.
// Every fold owns [0, quality_end) cursor values in its permutation order.
inline void OrderedAccumulate(float4 value, thread float4& sum, thread float4& error) {
    const float4 adjusted = value - error;
    const float4 next = sum + adjusted;
    error = (next - sum) - adjusted;
    sum = next;
}

kernel void PrepareOrderedRmseDerivatives(
    const device float* targets [[buffer(0)]], const device float* weights [[buffer(1)]],
    const device float* cursors [[buffer(2)]], const device uint* permutation [[buffer(3)]],
    const device uint4* folds [[buffer(4)]], device float2* derivatives [[buffer(5)]],
    constant OrderedParams& p [[buffer(6)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.y >= p.folds) return;
    const uint4 fold = folds[tid.y];
    if (tid.x >= fold.y) return;
    const uint row = permutation[tid.x], slot = fold.z + tid.x;
    derivatives[slot] = float2(weights[row] * (targets[row] - cursors[slot]), weights[row]);
}

// Derivatives are (weighted gradient, structure denominator). For Newton
// scoring the denominator is weighted curvature, not observation weight.
// Sampling factors belong to OCCURRENCES, not source documents. Keep the
// unsampled input for quality-noise statistics and leaf estimation.
kernel void ApplyOrderedBootstrap(const device float2* derivatives [[buffer(0)]],
    const device float* multipliers [[buffer(1)]], const device uint4* folds [[buffer(2)]],
    device float2* sampled [[buffer(3)]], constant OrderedParams& p [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    if (tid.y >= p.folds) return;
    const uint4 fold = folds[tid.y];
    if (tid.x >= fold.y) return;
    const uint slot = fold.z + tid.x;
    const float multiplier = p.test_only && tid.x < fold.x ? 1.0f : multipliers[slot - folds[0].z];
    sampled[slot] = derivatives[slot] * multiplier;
}

// One 256-thread group per candidate/leaf/fold. Output shape is
// [candidate, leaf, fold, side], with float4 = (estimate W, estimate G,
// quality W, quality G). This direct implementation is an arithmetic
// foundation; production should reuse partitioned histograms across leaves.
kernel void OrderedCandidateStatistics(const device uchar* bins [[buffer(0)]],
    const device uint* permutation [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    const device uint2* candidates [[buffer(3)]], const device uint* split_types [[buffer(4)]],
    const device float2* sampled [[buffer(5)]], const device uint4* folds [[buffer(6)]],
    device float4* statistics [[buffer(7)]], constant OrderedParams& p [[buffer(8)]],
    uint3 local [[thread_position_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    const uint candidate = group.x, leaf = group.y, fold_id = group.z;
    if (candidate >= p.candidates || leaf >= p.leaves || fold_id >= p.folds) return;
    threadgroup float4 left_scratch[256], right_scratch[256];
    const uint4 fold = folds[fold_id];
    const uint2 split = candidates[candidate];
    float4 left = 0, right = 0, left_error = 0, right_error = 0;
    for (uint position = tid; position < fold.y; position += 256) {
        const uint row = permutation[position];
        if (leaf_ids[row] != leaf) continue;
        const uint value = bins[row * p.features + split.x];
        const bool goes_right = split_types[candidate] ? value == split.y : value > split.y;
        const float2 derivative = sampled[fold.z + position];
        const float4 value_stats = position < fold.x
            ? float4(derivative.y, derivative.x, 0, 0)
            : float4(0, 0, derivative.y, derivative.x);
        if (goes_right) OrderedAccumulate(value_stats, right, right_error);
        else OrderedAccumulate(value_stats, left, left_error);
    }
    left_scratch[tid] = left;
    right_scratch[tid] = right;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            left_scratch[tid] += left_scratch[tid + stride];
            right_scratch[tid] += right_scratch[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        const uint output = ((candidate * p.leaves + leaf) * p.folds + fold_id) * 2;
        statistics[output] = left_scratch[0];
        statistics[output + 1] = right_scratch[0];
    }
}

// Lower scores/gains win; tie breaking by candidate index belongs to the
// existing winner reducer. Feature options float4 = categorical multiplier,
// feature penalty multiplier, already-scaled feature noise, reserved.
// score_function 0: Cosine/NewtonCosine; 1: legacy dynamic SolarL2.
// Current CUDA public options reject Ordered+SolarL2; do not expose this
// diagnostic variant as public CUDA parity. CUDA dynamic dispatch has no L2.
kernel void ScoreOrderedCandidates(const device float4* statistics [[buffer(0)]],
    const device uint2* candidates [[buffer(1)]], const device float4* feature_options [[buffer(2)]],
    device float2* scores [[buffer(3)]], constant OrderedParams& p [[buffer(4)]],
    uint candidate [[thread_position_in_grid]]) {
    if (candidate >= p.candidates) return;
    float score = 0.0f, norm = 1e-20f;
    for (uint leaf = 0; leaf < p.leaves; ++leaf) {
        float2 solar_score = 0.0f, quality_mass = 0.0f;
        for (uint fold = 0; fold < p.folds; ++fold) {
            for (uint side = 0; side < 2; ++side) {
                const float4 s = statistics[((candidate * p.leaves + leaf) * p.folds + fold) * 2 + side];
                const float lambda = p.normalize ? p.l2 * s.x : p.l2;
                const float mu = s.x > 0.0f ? s.y / (s.x + (p.score_function ? 1e-15f : lambda)) : 0.0f;
                if (!p.score_function) {
                    score += s.w * mu;
                    norm += s.z * mu * mu;
                } else {
                    solar_score[side] += -2.0f * mu * s.w + s.z * mu * mu;
                    quality_mass[side] += s.z;
                }
            }
        }
        if (p.score_function) {
            for (uint side = 0; side < 2; ++side) {
                if (quality_mass[side] > 2.0f)
                    score += solar_score[side] * (1.0f + 2.0f * log(quality_mass[side] + 1.0f));
            }
        }
    }
    if (!p.score_function) score = norm > 1e-15f ? -score / sqrt(norm) : FLT_MAX;
    const float4 options = feature_options[candidates[candidate].x];
    score = score * options.x + (p.score_function ? 0.0f : options.z);
    scores[candidate] = float2(score, (score - p.score_before) * options.y);
}

// Only quality rows contribute to Ordered score-noise variance. Count is
// row count, including zero-weight rows, not mass or duplicated prefix size.
kernel void OrderedQualityStatistics(const device float2* derivatives [[buffer(0)]],
    const device uint4* folds [[buffer(1)]], device float2* statistics [[buffer(2)]],
    constant OrderedParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint fold_id [[threadgroup_position_in_grid]]) {
    if (fold_id >= p.folds) return;
    threadgroup float scratch[256];
    const uint4 fold = folds[fold_id];
    float4 sum = 0.0f, error = 0.0f;
    for (uint position = fold.x + tid; position < fold.y; position += 256) {
        const float2 d = derivatives[fold.z + position];
        const float target = d.x > -1e-15f && d.x < 1e-15f ? 0.0f : d.x / (d.y + 1e-15f);
        OrderedAccumulate(float4(target * target * d.y, 0, 0, 0), sum, error);
    }
    scratch[tid] = sum.x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) statistics[fold_id] = float2(scratch[0], float(fold.y - fold.x));
}

// One RMSE leaf-estimation step. Use ORIGINAL unsampled derivatives, and
// reduce ONLY the estimate prefix. The separate full-model estimation task
// may use the same kernel with an extra descriptor (rows, rows, offset, 0).
// CUDA fold normalization divides G and H by TOTAL task weight, giving
// G/(H + l2*task_weight), distinct from split scoring's per-leaf scaling.
kernel void EstimateOrderedRmseLeaves(const device float2* derivatives [[buffer(0)]],
    const device uint* permutation [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    const device uint4* folds [[buffer(3)]], device float* leaf_values [[buffer(4)]],
    constant OrderedParams& p [[buffer(5)]], uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    const uint leaf = group.x, fold_id = group.y;
    if (leaf >= p.leaves || fold_id >= p.folds) return;
    threadgroup float4 scratch[256];
    const uint4 fold = folds[fold_id];
    float4 sum = 0.0f, error = 0.0f;
    for (uint position = tid; position < fold.x; position += 256) {
        const float2 d = derivatives[fold.z + position];
        const bool belongs = leaf_ids[permutation[position]] == leaf;
        OrderedAccumulate(float4(belongs ? d.x : 0.0f, belongs ? d.y : 0.0f, d.y, 0), sum, error);
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        const float4 s = scratch[0];
        const float denominator = s.y + p.l2 * (p.normalize ? s.z : 1.0f);
        leaf_values[fold_id * p.leaves + leaf] = s.y > 1e-20f && denominator > 0.0f ? s.x / denominator : 0.0f;
    }
}

kernel void ApplyOrderedFoldValues(device float* cursors [[buffer(0)]],
    const device uint* permutation [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    const device uint4* folds [[buffer(3)]], const device float* leaf_values [[buffer(4)]],
    constant OrderedParams& p [[buffer(5)]], uint2 tid [[thread_position_in_grid]]) {
    if (tid.y >= p.folds) return;
    const uint4 fold = folds[tid.y];
    if (tid.x >= fold.y) return;
    const uint leaf = leaf_ids[permutation[tid.x]];
    cursors[fold.z + tid.x] += p.learning_rate * leaf_values[tid.y * p.leaves + leaf];
}
)METAL";
