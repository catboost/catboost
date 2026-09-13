#pragma once

// Append after CBMMetalMulticlassScoresSource and CBMMetalGreedySource.
// CUDA greedy_subsets_searcher/kernel/compute_scores.cu retains double parent
// statistics, rounds each explicit right-child statistic to float, and adds
// the implicit MultiClass coordinate in double. The host supplies float-pair
// parent planes; histograms remain float32. Scalar greedy kernels are unchanged.
static const char* CBMMetalGreedyVectorScoresSource = R"METAL(
// Retain double's exponent range as well as expanded mantissas. CUDA can
// square a float gradient above 1e19, or retain tiny squares that a large LOO
// adjustment brings back into float range. Unscaled float-pair products lose
// those otherwise finite scores before the calcer's final float rounding.
struct GreedyVectorWide { float2 value; int exponent; };
inline GreedyVectorWide GreedyWide(float2 value, int exponent = 0) {
    if (value.x == 0) return {value, exponent};
    int shift;
    const float high = frexp(value.x, shift);
    return {float2(high, ldexp(value.y, -shift)), exponent + shift};
}
inline GreedyVectorWide GreedyWideAdd(GreedyVectorWide a, GreedyVectorWide b) {
    if (all(a.value == float2(0))) return b;
    if (all(b.value == float2(0))) return a;
    const int exponent = max(a.exponent, b.exponent);
    const float2 av = float2(ldexp(a.value.x, a.exponent-exponent), ldexp(a.value.y, a.exponent-exponent));
    const float2 bv = float2(ldexp(b.value.x, b.exponent-exponent), ldexp(b.value.y, b.exponent-exponent));
    return GreedyWide(VectorScoreAdd(av, bv), exponent);
}
inline GreedyVectorWide GreedyWideMultiply(GreedyVectorWide a, GreedyVectorWide b) {
    return GreedyWide(VectorScoreMultiply(a.value, b.value), a.exponent+b.exponent);
}
inline GreedyVectorWide GreedyWideDivide(GreedyVectorWide a, GreedyVectorWide b) {
    return GreedyWide(VectorScoreDivide(a.value, b.value), a.exponent-b.exponent);
}
inline float GreedyWideRound(GreedyVectorWide a) { return ldexp(VectorScoreRound(a.value), a.exponent); }
inline bool GreedyWeightAbove(float2 weight, float threshold) {
    return VectorScoreRound(VectorScoreAdd(weight, float2(-threshold, 0))) > 0;
}
struct GreedyVectorScore {
    float value;
    GreedyVectorWide numerator, denominator;
};
inline GreedyVectorScore EmptyGreedyVectorScore() {
    return {0, GreedyWide(float2(0)), GreedyWide(float2(1e-10f, 0))};
}
inline void AddGreedyVectorLeaf(thread GreedyVectorScore& score, float2 gradient,
                               float2 weight, constant GreedyParams& p) {
    const float w = VectorScoreRound(weight);
    if (p.score_function == 1) {
        if (!GreedyWeightAbove(weight, 0)) return;
        const float2 ridge = p.normalize ? VectorScoreMultiply(float2(p.l2, 0), weight) : float2(p.l2, 0);
        const auto g = GreedyWide(gradient), mu = GreedyWideDivide(g, GreedyWide(VectorScoreAdd(weight, ridge)));
        score.numerator = GreedyWideAdd(score.numerator, GreedyWideMultiply(g, mu));
        score.denominator = GreedyWideAdd(score.denominator, GreedyWideMultiply(GreedyWideMultiply(GreedyWide(weight), mu), mu));
    } else {
        if (!GreedyWeightAbove(weight, p.score_function == 0 || p.score_function == 4 ? 1e-20f : 0.0f)) return;
        float2 denominator = weight, adjustment = float2(1, 0);
        if (p.score_function == 0) denominator = VectorScoreAdd(weight, float2(p.l2, 0));
        else if (p.score_function == 4) {
            const float2 argument = VectorScoreAdd(weight, float2(1, 0));
            const float2 logarithm = float2(log(argument.x), argument.y / argument.x);
            adjustment = VectorScoreAdd(float2(1, 0), VectorScoreMultiply(float2(2, 0), logarithm));
        } else if (p.score_function == 5) {
            float factor = GreedyWeightAbove(weight, 1) ? VectorScoreRound(VectorScoreDivide(weight, VectorScoreAdd(weight, float2(-1, 0)))) : 0;
            factor *= factor; adjustment = float2(factor, 0);
        } else if (p.score_function == 6) {
            if (!GreedyWeightAbove(weight, 2)) return;
            float2 numerator, divisor;
            if (w <= 4) {
                numerator = VectorScoreMultiply(weight, VectorScoreAdd(weight, float2(-2, 0)));
                divisor = VectorScoreAdd(VectorScoreMultiply(weight, VectorScoreAdd(weight, float2(-3, 0))), float2(1, 0));
            } else {
                const float2 inverse = VectorScoreDivide(float2(1, 0), weight);
                numerator = VectorScoreAdd(float2(1, 0), VectorScoreMultiply(float2(-2, 0), inverse));
                divisor = VectorScoreAdd(VectorScoreAdd(float2(1, 0),
                    VectorScoreMultiply(float2(-3, 0), inverse)), VectorScoreMultiply(inverse, inverse));
            }
            adjustment = float2(VectorScoreRound(VectorScoreDivide(numerator, divisor)), 0);
        }
        if (gradient.x == 0 || adjustment.x == 0 || !isfinite(score.value)) return;
        const auto term = GreedyWideMultiply(GreedyWideMultiply(GreedyWide(-gradient),
            GreedyWideDivide(GreedyWide(gradient), GreedyWide(denominator))), GreedyWide(adjustment));
        // L2/Solar/LOO/Sat store Score as float after each AddLeaf in CUDA.
        score.value = GreedyWideRound(GreedyWideAdd(GreedyWide(float2(score.value, 0)), term));
    }
}
inline float FinishGreedyVectorScore(GreedyVectorScore score, float noise, constant GreedyParams& p) {
    if (p.score_function != 1) return score.value;
    const int odd = score.denominator.exponent & 1;
    const float2 argument = score.denominator.value * (odd ? 2.0f : 1.0f);
    const float high = sqrt(argument.x);
    const float2 residual = VectorScoreAdd(argument, -VectorScoreMultiply(float2(high, 0), float2(high, 0)));
    const float2 root = VectorScoreAdd(float2(high, 0), float2(VectorScoreRound(residual) / (2 * high), 0));
    return -GreedyWideRound(GreedyWideDivide(score.numerator,
        GreedyWide(root, (score.denominator.exponent-odd)/2))) + noise;
}

kernel void FindGreedyVectorSplitWinners(
    const device float* sums [[buffer(0)]], const device float* weights [[buffer(1)]],
    const device float2* leaf_sums [[buffer(2)]], const device float2* leaf_weights [[buffer(3)]],
    const device uint* features [[buffer(4)]], const device uint* bins [[buffer(5)]],
    const device uchar* types [[buffer(6)]], const device uint* feature_offsets [[buffer(7)]],
    const device float* feature_weights [[buffer(8)]], const device float* feature_noise [[buffer(9)]],
    device GreedySplit* winners [[buffer(10)]], constant GreedyParams& p [[buffer(11)]],
    uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    const uint leaf = group.y;
    GreedySplit best = EmptyGreedySplit(leaf);
    uint bad = 0;
    for (uint candidate = group.x * 256 + tid; candidate < p.candidates; candidate += p.score_groups * 256) {
        const uint feature = features[candidate];
        if (feature < p.feature_begin || feature - p.feature_begin >= p.features) continue;
        const ulong cell = ulong(leaf) * p.total_bins + feature_offsets[feature-p.feature_begin] + bins[candidate];
        const float2 parent_weight = leaf_weights[leaf];
        const float selected_weight = max(weights[cell], 0.0f);
        const float other_weight = max(VectorScoreRound(VectorScoreAdd(parent_weight, float2(-selected_weight, 0))), 0.0f);
        const bool zero_child = selected_weight < 1e-20f || other_weight < 1e-20f;
        auto before = EmptyGreedyVectorScore(), after = EmptyGreedyVectorScore();
        float2 missing_selected = float2(0), missing_parent = float2(0);
        for (uint k = 0; k < p.dimensions + p.multiclass_optimization; ++k) {
            float2 selected, parent, other;
            if (k < p.dimensions) {
                selected = float2(sums[ulong(k)*p.histogram_stride+cell], 0);
                parent = leaf_sums[ulong(k)*p.leaf_stride+leaf];
                other = float2(VectorScoreRound(VectorScoreAdd(parent, -selected)), 0);
                missing_selected = VectorScoreAdd(missing_selected, -selected);
                missing_parent = VectorScoreAdd(missing_parent, -parent);
            } else {
                selected = missing_selected; parent = missing_parent;
                other = VectorScoreAdd(parent, -selected);
            }
            AddGreedyVectorLeaf(after, selected, float2(selected_weight, 0), p);
            AddGreedyVectorLeaf(after, other, float2(other_weight, 0), p);
            AddGreedyVectorLeaf(before, parent, parent_weight, p);
        }
        const float noise = feature_noise[feature];
        const float gain = zero_child ? 0 : (FinishGreedyVectorScore(after, noise, p) -
            FinishGreedyVectorScore(before, noise, p)) * feature_weights[feature];
        bad |= uint(!isfinite(gain));
        GreedySplit value = {candidate, feature, bins[candidate], uint(types[candidate]), gain,
            uint(isfinite(gain)), 0, leaf};
        if (BetterGreedySplit(value, best)) best = value;
    }
    threadgroup GreedySplit partial[256];
    threadgroup uint errors[256];
    partial[tid] = best; errors[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            if (BetterGreedySplit(partial[tid+stride], partial[tid])) partial[tid] = partial[tid+stride];
            errors[tid] |= errors[tid+stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) { auto result = partial[0]; result.error = errors[0]; winners[leaf*p.score_groups+group.x] = result; }
}
)METAL";
