#pragma once

// Append after CBMMetalSource and CBMMetalStreamingScoreSource. Reuses their
// KernelParams, SplitState, ScoreTileParams and canonical float-pair helpers;
// no alternate numerical helper or ABI definition is introduced here.
//
// Buffers 0..10 match FindTileSplitWinners. candidate_active is uchar[C] at
// buffer 11: zero disables the candidate; any nonzero byte enables it.
// KernelParams (96 bytes) is at 12 and ScoreTileParams (32 bytes) is at 13.
// Inactive candidates are skipped BEFORE their feature/bin metadata or any
// histogram, noise, or penalty is accessed. The host may retain inactive
// placeholders while preparing a new feature, but must validate every active
// candidate against the committed dataset and its compact feature span.
//
// Candidate ids and feature ids remain global. Dispatch p.reserved2 groups
// of exactly 256 threads; use existing ReduceSplitWinners and, across feature
// tiles, MergeTileSplitWinner. All-inactive tiles return neutral partials with
// valid=0, score=gain=INFINITY and no nonfinite error. Only active candidates
// contribute errors. Enable a new mask after all of its metadata/histograms
// are ready, using ordered commands or a completed CPU/GPU synchronization.
//
// Numerical operations intentionally match the canonical tiled scorer:
// L2/NewtonL2 (0/2), Cosine/NewtonCosine (1/3), SolarL2 (4), LOOL2 (5),
// feature noise, raw-score penalties, gain penalties and global-id tie breaks.
static const char* CBMMetalDynamicScoreSource = R"METAL(

kernel void FindDynamicTileSplitWinners(
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
        float score = 0;
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
                    score = AddL2ScoreLeaf(score, sum, weight, p.l2);
                } else {
                    AddCosineScoreLeaf(numerator, denominator, sum, weight, p.l2);
                }
            }
        }
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
)METAL";
