#pragma once

// Appended after the common scalar types and multiclass objective kernels.
static const char* CBMMetalMulticlassTrainingSource = R"METAL(
struct MulticlassTrainingParams {
    uint dimensions, histogram_stride, leaf_stride, reserved;
};

inline float2 MulticlassPairAdd(float2 left, float2 right) {
    float high = left.x, low = left.y;
    MulticlassAdd(high, low, right.x); MulticlassAdd(high, low, right.y);
    return float2(high, low);
}
inline float MulticlassPairRound(float2 value) { return value.x + value.y; }

// CUDA TL2ScoreCalcer widens g, w and lambda, then rounds Score to float
// after adding EACH leaf term. Preserve those intermediates using float pairs.
inline float MulticlassAddL2Leaf(float score, float2 gradient, float weight, float l2) {
    if (weight <= 1e-20f) return score;
    const float2 denominator = MulticlassPairAdd(float2(weight, 0), float2(l2, 0));
    const float quotient_high = gradient.x / denominator.x;
    float residual = fma(-quotient_high, denominator.x, gradient.x);
    residual += gradient.y - quotient_high * denominator.y;
    const float quotient_low = residual / denominator.x;
    const float product_high = gradient.x * quotient_high;
    float product_low = fma(gradient.x, quotient_high, -product_high);
    product_low += gradient.x * quotient_low + gradient.y * quotient_high;
    product_low += gradient.y * quotient_low;
    return MulticlassPairRound(MulticlassPairAdd(float2(score, 0), -float2(product_high, product_low)));
}

// Retain the compensated parent statistics, as CUDA's double partStats does.
// Histograms remain float32, and active right gradients are rounded to float
// after subtracting the selected histogram sum from the wider parent sum.
kernel void MulticlassCollectPartitionStatistics(const device float4* partials [[buffer(0)]],
                                                 device float2* sums [[buffer(1)]],
                                                 device float2* weights [[buffer(2)]],
                                                 constant KernelParams& p [[buffer(3)]],
                                                 uint tid [[thread_position_in_threadgroup]],
                                                 uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float4 high[256], low[256];
    const uint cell = 2 * (leaf * p.reserved0 + tid);
    high[tid] = tid < p.reserved0 ? partials[cell] : float4(0);
    low[tid] = tid < p.reserved0 ? partials[cell + 1] : float4(0);
    ObjectiveReduceExpansions(high, low, tid);
    if (tid == 0) {
        sums[leaf] = float2(high[0].x, low[0].x);
        weights[leaf] = float2(high[0].z, low[0].z);
    }
}

// CUDA greedy_subsets_searcher/kernel/compute_scores.cu scores every class
// against observation weights. MultiClass reconstructs the missing gradient
// as minus the sum of active-class gradients, including it in every scorer.
kernel void MulticlassFindSplitWinners(
    const device float* sums [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device float2* leaf_sums [[buffer(2)]],
    const device float2* leaf_weights [[buffer(3)]],
    const device uint* features [[buffer(4)]],
    const device uint* bins [[buffer(5)]],
    const device uchar* types [[buffer(6)]],
    device SplitState* winners [[buffer(7)]],
    const device float* feature_noise [[buffer(8)]],
    const device float* feature_weights [[buffer(9)]],
    constant KernelParams& p [[buffer(10)]],
    constant MulticlassTrainingParams& m [[buffer(11)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    float best_score = INFINITY;
    float best_raw_score = INFINITY;
    uint best_index = 0xffffffffu, bad = 0;
    for (uint candidate = group * 256 + tid; candidate < p.candidates;
         candidate += p.reserved2 * 256) {
        float score = 0, numerator = 0, denominator = 1e-10f;
        for (uint leaf = 0; leaf < p.leaves; ++leaf) {
            const uint cell = (leaf * p.features + features[candidate]) * p.bins + bins[candidate];
            const float selected_weight = weights[cell];
            const float other_weight = max(p.score_function != 1
                ? MulticlassPairRound(MulticlassPairAdd(leaf_weights[leaf], float2(-selected_weight, 0)))
                : MulticlassPairRound(leaf_weights[leaf]) - selected_weight, 0.0f);
            float missing_selected = 0, missing_other = 0;
            float2 missing_selected_pair = float2(0), missing_parent_pair = float2(0);
            for (uint k = 0; k < m.dimensions + uint(p.objective == 0); ++k) {
                float selected, other;
                float2 selected_pair, other_pair;
                if (k < m.dimensions) {
                    selected = sums[k * m.histogram_stride + cell];
                    const float2 parent = leaf_sums[k * m.leaf_stride + leaf];
                    other = p.score_function != 1
                        ? MulticlassPairRound(MulticlassPairAdd(parent, float2(-selected, 0)))
                        : MulticlassPairRound(parent) - selected;
                    missing_selected -= selected; missing_other -= other;
                    missing_selected_pair = MulticlassPairAdd(missing_selected_pair, float2(-selected, 0));
                    missing_parent_pair = MulticlassPairAdd(missing_parent_pair, -parent);
                    selected_pair = float2(selected, 0); other_pair = float2(other, 0);
                } else {
                    selected = missing_selected; other = missing_other;
                    selected_pair = missing_selected_pair;
                    other_pair = MulticlassPairAdd(missing_parent_pair, -missing_selected_pair);
                }
                for (uint side = 0; side < 2; ++side) {
                    const float gradient = side ? other : selected;
                    const float weight = side ? other_weight : selected_weight;
                    if (p.score_function == 0) {
                        score = MulticlassAddL2Leaf(score, side ? other_pair : selected_pair, weight, p.l2);
                    } else if (p.score_function == 1) {
                        const float value = weight > 0 ? gradient / (weight + p.l2) : 0;
                        numerator += gradient * value;
                        denominator += weight * value * value;
                    } else {
                        score = VectorScoreAddLeaf(score, side ? other_pair : selected_pair, weight, p.score_function);
                    }
                }
            }
        }
        float raw_score = score;
        if (p.score_function == 1) {
            // CUDA symmetric greedy search compares Gain, subtracting the
            // empty pre-split calculator with the identical feature draw.
            // Retain both float operations: the noise cancels except rounding.
            const float noise = feature_noise[features[candidate]];
            raw_score = -numerator / sqrt(denominator) + noise;
            score = raw_score - noise;
        }
        // Greedy CUDA multiplies gain by its dynamic feature weight, without
        // subtracting the scalar path's previous weighted depth score.
        score *= feature_weights[features[candidate]];
        bad |= uint(!isfinite(score) || !isfinite(raw_score));
        if (isfinite(score) && (score < best_score || (score == best_score && candidate < best_index))) {
            best_score = score; best_raw_score = raw_score; best_index = candidate;
        }
    }
    threadgroup float scores[256], raw_scores[256];
    threadgroup uint indices[256], errors[256];
    scores[tid] = best_score; indices[tid] = best_index; errors[tid] = bad;
    raw_scores[tid] = best_raw_score;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            uint other = tid + stride;
            if (scores[other] < scores[tid] || (scores[other] == scores[tid] && indices[other] < indices[tid])) {
                scores[tid] = scores[other]; raw_scores[tid] = raw_scores[other]; indices[tid] = indices[other];
            }
            errors[tid] |= errors[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        uint index = indices[0];
        SplitState result = {index, 0, 0, 0, raw_scores[0], uint(index != 0xffffffffu), errors[0], scores[0]};
        if (result.valid) { result.feature = features[index]; result.bin = bins[index]; result.type = types[index]; }
        winners[group] = result;
    }
}

kernel void MulticlassInitializeTree(device uint* leaf_ids [[buffer(0)]],
                                     device float* raw_values [[buffer(1)]],
                                     constant MulticlassParams& p [[buffer(2)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index < p.rows) leaf_ids[index] = 0;
    if (index < p.leaves * MulticlassDimension(p)) raw_values[index] = 0;
}

kernel void MulticlassAccumulateDirections(const device float* directions [[buffer(0)]],
                                           device float* raw_values [[buffer(1)]],
                                           constant MulticlassParams& p [[buffer(2)]],
                                           uint index [[thread_position_in_grid]]) {
    if (index < p.leaves * MulticlassDimension(p)) raw_values[index] += directions[index];
}

// Rebuild from the ensemble cursor instead of successively adding row deltas;
// this matches CUDA's leaf point optimization and avoids accumulating rounding.
kernel void MulticlassBuildCursor(const device float* base [[buffer(0)]],
                                  const device float* raw_values [[buffer(1)]],
                                  const device uint* leaf_ids [[buffer(2)]],
                                  device float* cursor [[buffer(3)]],
                                  constant MulticlassParams& p [[buffer(4)]],
                                  constant float& step [[buffer(5)]],
                                  uint index [[thread_position_in_grid]]) {
    uint dimensions = MulticlassDimension(p);
    if (index >= p.rows * dimensions) return;
    uint row = index % p.rows, k = index / p.rows;
    cursor[index] = base[index] + step * raw_values[leaf_ids[row] * dimensions + k];
}

// Keep the original full raw gauge separately: subtracting a large last-class
// anchor and adding it back can destroy a small initial class value. Published
// predictions receive only committed tree deltas, without that cancellation.
kernel void MulticlassBuildPublishedCursor(const device float* base [[buffer(0)]],
                                           const device float* raw_values [[buffer(1)]],
                                           const device uint* leaf_ids [[buffer(2)]],
                                           device float* cursor [[buffer(3)]],
                                           constant MulticlassParams& p [[buffer(4)]],
                                           constant float& step [[buffer(5)]],
                                           uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.classes) return;
    uint row = index / p.classes, k = index % p.classes;
    uint dimensions = MulticlassDimension(p);
    cursor[index] = k < dimensions ? base[index] + step * raw_values[leaf_ids[row] * dimensions + k] : base[index];
}
)METAL";
