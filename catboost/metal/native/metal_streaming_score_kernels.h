#pragma once

// Append after CBMMetalSource, which defines the 96-byte KernelParams and
// 32-byte SplitState. ScoreTileParams is a separate 32-byte constant block:
// {feature_begin,feature_end,total_bins,reserved0,reserved1,reserved2,
//  reserved3,reserved4}; all reserved fields are zero.
//
// KernelParams and candidate arrays retain GLOBAL feature/candidate indexing.
// A tile covers [feature_begin,feature_end), with tile-local feature_offsets
// of length feature_end-feature_begin+1, starting at 0 and ending total_bins.
// Every candidate belonging to this tile must have bin < its feature span.
// Sums/weights each contain leaves*total_bins floats; use checked host sizes
// and enforce the aggregate memory budget before dispatching. Empty feature
// ranges or ranges with no candidates produce neutral partial winners.
//
// Dispatch FindTileSplitWinners with p.reserved2 groups of 256 threads, then
// unchanged ReduceSplitWinners with one group of 256. Score groups must be in
// [1,256]. Reduce the first tile directly into the global winner; reduce each
// later tile into separate tile-winner storage, then MergeTileSplitWinner
// with one thread. Tiles may arrive in any order; use the same global feature
// noise buffer throughout one depth. Cover every candidate feature to obtain
// the same global winner/error state as scoring the entire feature cache.
// Histogram construction, scoring and winner merges must be ordered before
// reusing tile scratch. The existing split score arithmetic remains float32.
static const char* CBMMetalStreamingScoreSource = R"METAL(

struct ScoreTileParams {
    uint feature_begin, feature_end, total_bins, reserved0;
    uint reserved1, reserved2, reserved3, reserved4;
};

// CUDA FindOptimalSplitSingleFoldImpl and {TL2ScoreCalcer,TCosineScoreCalcer},
// matching current FindSplitWinners operation order. The only scoring-path
// changes are the tile membership check and compact local histogram address.
// Candidates are never reordered or renumbered; noise uses global features.
// Score ids 0/2 use L2 and 1/3 use Cosine; the runtime supplies observation
// weights or Hessians for the selected family. Noise applies only to Cosine.
kernel void FindTileSplitWinners(
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
        float score = 0;
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

// Threads: 1. Exactly the comparison/error combination used by
// ReduceSplitWinners. Neutral losing tiles must still propagate error flags.
// The incoming tile winner must already have been reduced on the GPU.
kernel void MergeTileSplitWinner(device SplitState* global_winner [[buffer(0)]],
                                  const device SplitState* tile_winner [[buffer(1)]],
                                  constant KernelParams& p [[buffer(2)]],
                                  uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    SplitState result = global_winner[0];
    const SplitState other = tile_winner[0];
    const uint bad = result.reserved0 | other.reserved0;
    if (other.gain < result.gain || (other.gain == result.gain && other.index < result.index))
        result = other;
    result.reserved0 = bad;
    global_winner[0] = result;
}
)METAL";
