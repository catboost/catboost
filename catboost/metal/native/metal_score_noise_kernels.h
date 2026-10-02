#pragma once

// Concatenate after CBMMetalBootstrapSource, which supplies BootstrapParams
// and the CUDA-derived multiply-with-carry/Box-Muller RNG helpers.
static const char* CBMMetalScoreNoiseSource = R"METAL(

// random_score_helper.h::ComputeStdDev first calls DivideVector with
// SkipZeroesOnDivide=true (transform.cpp), then weighted DotProduct. Preserve
// operators.cuh::ZeroAwareDivide's strict tiny-gradient test and +1e-15.
inline float ScoreNoiseWeakTarget(float gradient, float weight) {
    return gradient > -1e-15f && gradient < 1e-15f
        ? 0.0f : gradient / (weight + 1e-15f);
}

// Use ORIGINAL weighted gradients and ORIGINAL observation weights before
// bootstrap. Partial outputs already divide by ROW COUNT, not sum(weights).
// Sum the bounded partial array in host double, sqrt, then multiply by
// random_strength * logistic(log(rows)-absolute_iteration*learning_rate).
// The resulting scale stays fixed throughout one tree's structure search.
kernel void ReduceScoreNoiseStatistics(const device float* gradients [[buffer(0)]],
                                       const device float* weights [[buffer(1)]],
                                       device float* partials [[buffer(2)]],
                                       constant BootstrapParams& p [[buffer(3)]],
                                       uint tid [[thread_position_in_threadgroup]],
                                       uint group [[threadgroup_position_in_grid]],
                                       uint groups [[threadgroups_per_grid]]) {
    threadgroup float scratch[256];
    float sum = 0.0f, correction = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float weight = weights[row];
        if (weight == 0.0f) continue;
        const float weak_target = ScoreNoiseWeakTarget(gradients[row], weight);
        // Scale before squaring to avoid avoidable float32 overflow. CUDA
        // performs a signed weighted dot product with wider intermediates;
        // Combination's YetiRank coefficients can make row weights negative.
        const float normalized = weak_target * sqrt(abs(weight) / float(p.rows));
        const float value = copysign(normalized * normalized, weight) - correction;
        const float next = sum + value;
        correction = (next - sum) - value;
        sum = next;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = scratch[0];
}

// score_calcers.cuh::TCosineScoreCalcer adds one normal draw per FEATURE;
// all threshold candidates for that feature reuse the same perturbation.
// Set rows=feature_count, iteration=absolute tree index, stream=depth+1 to
// separate every depth from bootstrap's stream0, noise_scale=tree scale.
// Metal stream initialization is intentionally independent of CUDA launches.
kernel void GenerateScoreFeatureNoise(device float* noise [[buffer(0)]],
                                      constant BootstrapParams& p [[buffer(1)]],
                                      uint feature [[thread_position_in_grid]]) {
    if (feature < p.rows) noise[feature] = BootstrapNormalForItem(feature, p) * p.noise_scale;
}
)METAL";
