#pragma once

// Additive option kernels retain the default score/leaf entrypoints and their
// bindings. Append after objective, backtracking and tiled scoring sources.
static const char* CBMMetalRegularizationSource = R"METAL(
struct ScoreRegularizationParams {
    uint normalize;
    float meta_exponent;
    uint per_feature, reserved;
};
inline float AddMetaL2ScoreLeaf(float score, float sum, float weight, float l2, float exponent) {
    if (exponent == 1.0f) return AddL2ScoreLeaf(score, sum, weight, l2);
    if (weight <= 1e-20f) return score;
    const float2 regularized = ScorePairAdd(float2(weight, 0), float2(l2, 0));
    const float2 mean = ScorePairDivide(float2(sum, 0), regularized);
    const float2 term = ScorePairMultiply(float2(sum, 0), mean);
    if (all(term == 0.0f)) return score;
    const float scaled = ScorePairRound(ScorePairDivide(term, float2(weight, 0)));
    const float transformed = pow(abs(scaled), exponent);
    return ScorePairRound(ScorePairAdd(float2(score, 0),
        -ScorePairMultiply(float2(transformed, 0), float2(weight, 0))));
}
inline void AddNormalizedCosineScoreLeaf(thread float2& numerator, thread float2& denominator,
                                         float sum, float weight, float l2, bool normalize) {
    if (!normalize) { AddCosineScoreLeaf(numerator, denominator, sum, weight, l2); return; }
    if (weight <= 0.0f) return;
    const float2 regularized = ScorePairAdd(float2(weight, 0),
        ScorePairMultiply(float2(weight, 0), float2(l2, 0)));
    const float2 mean = ScorePairDivide(float2(sum, 0), regularized);
    numerator = ScorePairAdd(numerator, ScorePairMultiply(float2(sum, 0), mean));
    denominator = ScorePairAdd(denominator,
        ScorePairMultiply(ScorePairMultiply(float2(weight, 0), mean), mean));
}

kernel void FindSplitWinnersRegularized(const device float* sums [[buffer(0)]],
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
                             constant ScoreRegularizationParams& regularization [[buffer(12)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint group [[threadgroup_position_in_grid]]) {
    float best_score = INFINITY, best_raw_score = INFINITY;
    uint best_index = 0xffffffffu;
    uint bad = 0;
    for (uint candidate = group * 256 + tid; candidate < p.candidates;
         candidate += p.reserved2 * 256) {
        float score = 0, default_score = 0;
        const uint choices = regularization.per_feature ? uint(feature_noise[features[candidate]]) : 2u;
        const float exponent = choices == 1u ? 1.0f : regularization.meta_exponent;
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
                    score = AddMetaL2ScoreLeaf(score, sum, weight, p.l2, exponent);
                    if (choices == 3u) default_score = AddL2ScoreLeaf(default_score, sum, weight, p.l2);
                } else {
                    AddNormalizedCosineScoreLeaf(numerator, denominator, sum, weight, p.l2, regularization.normalize != 0);
                }
            }
        }
        if ((p.score_function == 0 || p.score_function == 2) && choices == 3u) score = min(score, default_score);
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

kernel void FindTileSplitWinnersRegularized(
    const device float* sums [[buffer(0)]],
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
    constant ScoreTileParams& tile [[buffer(12)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    float best_score = INFINITY, best_raw_score = INFINITY;
    uint best_index = 0xffffffffu;
    uint bad = 0;
    for (uint candidate = group * 256 + tid; candidate < p.candidates;
         candidate += p.reserved2 * 256) {
        const uint feature = features[candidate];
        if (feature < tile.feature_begin || feature >= tile.feature_end) continue;
        float score = 0, default_score = 0;
        const uint choices = tile.reserved2 ? uint(feature_noise[features[candidate]]) : 2u;
        const float exponent = choices == 1u ? 1.0f : as_type<float>(tile.reserved1);
        float2 numerator = 0.0f, denominator = float2(1e-10f, 0);
        for (uint leaf = 0; leaf < p.leaves; ++leaf) {
            const ulong cell = ulong(leaf) * tile.total_bins
                + feature_offsets[feature - tile.feature_begin] + bins[candidate];
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
                    score = AddMetaL2ScoreLeaf(score, sum, weight, p.l2, exponent);
                    if (choices == 3u) default_score = AddL2ScoreLeaf(default_score, sum, weight, p.l2);
                } else {
                    AddNormalizedCosineScoreLeaf(numerator, denominator, sum, weight, p.l2, tile.reserved0 != 0);
                }
            }
        }
        if ((p.score_function == 0 || p.score_function == 2) && choices == 3u) score = min(score, default_score);
        if (p.score_function == 1 || p.score_function == 3)
            score = FinalizeCosineScore(numerator, denominator) + feature_noise[feature];
        score *= feature_penalties[feature].x;
        const float gain = (score - p.score_before_split) * feature_penalties[feature].y;
        bad |= uint(!isfinite(score) || !isfinite(gain));
        if (isfinite(score) && isfinite(gain) && (gain < best_score || (gain == best_score && candidate < best_index))) {
            best_score = gain; best_raw_score = score;
            best_index = candidate;
        }
    }
    threadgroup float local_scores[256], local_raw_scores[256];
    threadgroup uint local_indices[256], local_bad[256];
    local_scores[tid] = best_score; local_raw_scores[tid] = best_raw_score;
    local_indices[tid] = best_index;
    local_bad[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint other = tid + stride;
            if (local_scores[other] < local_scores[tid] ||
                (local_scores[other] == local_scores[tid] && local_indices[other] < local_indices[tid])) {
                local_scores[tid] = local_scores[other]; local_raw_scores[tid] = local_raw_scores[other];
                local_indices[tid] = local_indices[other];
            }
            local_bad[tid] |= local_bad[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const uint index = local_indices[0];
        SplitState result = {index, 0, 0, 0, local_raw_scores[0], uint(index != 0xffffffffu), local_bad[0], local_scores[0]};
        if (result.valid) {
            result.feature = features[index];
            result.bin = bins[index];
            result.type = types[index];
        }
        winners[group] = result;
    }
}

kernel void FindDynamicTileSplitWinnersRegularized(
    const device float* sums [[buffer(0)]],
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
    const device uchar* candidate_active [[buffer(11)]],
    constant KernelParams& p [[buffer(12)]],
    constant ScoreTileParams& tile [[buffer(13)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    float best_score = INFINITY, best_raw_score = INFINITY;
    uint best_index = 0xffffffffu;
    uint bad = 0;
    for (uint candidate = group * 256 + tid; candidate < p.candidates;
         candidate += p.reserved2 * 256) {
        if (candidate_active[candidate] == 0) continue;
        const uint feature = features[candidate];
        if (feature < tile.feature_begin || feature >= tile.feature_end) continue;
        float score = 0, default_score = 0;
        const uint choices = tile.reserved2 ? uint(feature_noise[features[candidate]]) : 2u;
        const float exponent = choices == 1u ? 1.0f : as_type<float>(tile.reserved1);
        float2 numerator = 0.0f, denominator = float2(1e-10f, 0);
        for (uint leaf = 0; leaf < p.leaves; ++leaf) {
            const ulong cell = ulong(leaf) * tile.total_bins
                + feature_offsets[feature - tile.feature_begin] + bins[candidate];
            const float selected_sum = sums[cell];
            const float selected_weight = weights[cell];
            const float other_sum = leaf_sums[leaf] - selected_sum;
            const float other_weight = max(leaf_weights[leaf] - selected_weight, 0.0f);
            // CUDA AddLeaf order is selected histogram, then complement,
            // including one-hot splits whose model routing is reversed.
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
                    score = AddMetaL2ScoreLeaf(score, sum, weight, p.l2, exponent);
                    if (choices == 3u) default_score = AddL2ScoreLeaf(default_score, sum, weight, p.l2);
                } else {
                    AddNormalizedCosineScoreLeaf(numerator, denominator, sum, weight, p.l2, tile.reserved0 != 0);
                }
            }
        }
        if ((p.score_function == 0 || p.score_function == 2) && choices == 3u) score = min(score, default_score);
        if (p.score_function == 1 || p.score_function == 3)
            score = FinalizeCosineScore(numerator, denominator) + feature_noise[feature];
        score *= feature_penalties[feature].x;
        const float gain = (score - p.score_before_split) * feature_penalties[feature].y;
        bad |= uint(!isfinite(score) || !isfinite(gain));
        if (isfinite(score) && isfinite(gain) && (gain < best_score || (gain == best_score && candidate < best_index))) {
            best_score = gain; best_raw_score = score;
            best_index = candidate;
        }
    }
    threadgroup float local_scores[256], local_raw_scores[256];
    threadgroup uint local_indices[256], local_bad[256];
    local_scores[tid] = best_score; local_raw_scores[tid] = best_raw_score;
    local_indices[tid] = best_index;
    local_bad[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint other = tid + stride;
            if (local_scores[other] < local_scores[tid] ||
                (local_scores[other] == local_scores[tid] && local_indices[other] < local_indices[tid])) {
                local_scores[tid] = local_scores[other]; local_raw_scores[tid] = local_raw_scores[other];
                local_indices[tid] = local_indices[other];
            }
            local_bad[tid] |= local_bad[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const uint index = local_indices[0];
        SplitState result = {index, 0, 0, 0, local_raw_scores[0], uint(index != 0xffffffffu), local_bad[0], local_scores[0]};
        if (result.valid) {
            result.feature = features[index];
            result.bin = bins[index];
            result.type = types[index];
        }
        winners[group] = result;
    }
}

kernel void EstimateRegularizedNewtonLeafValues(const device float4* partials [[buffer(0)]],
                                     device float* raw_values [[buffer(1)]],
                                     device float* output_weights [[buffer(2)]],
                                     constant KernelParams& p [[buffer(3)]],
                                     constant BacktrackingParams& b [[buffer(4)]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float4 high_scratch[256];
    threadgroup float4 low_scratch[256];
    if (leaf >= p.leaves) return;
    const uint tile_count = ObjectiveLeafTileCount(p);
    float3 high = float3(0.0f);
    float3 low = float3(0.0f);
    for (uint tile = tid; tile < tile_count; tile += 256) {
        const uint partial = 2 * (leaf * tile_count + tile);
        ObjectiveMergeExpansion(high, low, partials[partial].xyz,
                                 partials[partial + 1].xyz);
    }
    high_scratch[tid] = float4(high, 0.0f);
    low_scratch[tid] = float4(low, 0.0f);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) {
        const float3 statistics = high_scratch[0].xyz + low_scratch[0].xyz;
        float gradient = statistics.x;
        const float hessian = statistics.y;
        const float weight = statistics.z;
        output_weights[leaf] = weight;
        float diagonal = p.leaf_method == 1 ? weight : hessian;
        if (b.normalize) { gradient /= p.total_weight; diagonal /= p.total_weight; }
        if (b.add_ridge) gradient -= p.l2 * raw_values[leaf];
        diagonal += p.l2;
        // QuerySoftMax permits signed curvature parameters. Preserve its
        // literal Hessian when the regularized Newton diagonal is positive.
        // Combination's negated YetiRank coefficient can also produce a
        // nonpositive diagonal; CUDA leaves its move direction at zero.
        const bool grouped = p.objective == 12 || p.objective == 13 || p.objective == 19;
        if (!all(isfinite(statistics)) || (!grouped && hessian < 0.0f) || weight < 0.0f
            || (grouped && p.objective != 19 && p.leaf_method == 0 && weight >= 1e-20f && diagonal <= 0.0f)
            || !isfinite(diagonal) || !isfinite(raw_values[leaf])) {
            // Preserve a detectable error for the runtime's finite-output
            // checks, instead of silently selecting a finite replacement.
            raw_values[leaf] = ObjectiveInvalidValue();
        } else if (weight < 1e-20f) {
            raw_values[leaf] = 0.0f;
        } else if (diagonal > 0.0f) {
            const float next = raw_values[leaf] + gradient / (diagonal + 1e-20f);
            raw_values[leaf] = isfinite(next) ? next : ObjectiveInvalidValue();
        }
    }
}
)METAL";
