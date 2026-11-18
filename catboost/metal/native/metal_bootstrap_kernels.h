#pragma once

// CUDA-derived object bootstrap and RNG primitives. Concatenate this source
// before any score kernel that calls BootstrapNormalForItem. It is standalone
// MSL, with no dependency on the training KernelParams structure.
static const char* CBMMetalBootstrapSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct BootstrapParams {
    uint rows, type, seed_low, seed_high;
    uint iteration, stream, reserved0, reserved1;
    float temperature, subsample, mvs_lambda, noise_scale;
};

// The multiply-with-carry step is random_gen.cuh::AdvanceSeed verbatim in
// uint arithmetic. CUDA owns mutable per-launch-thread seeds; Metal derives
// one stream per object and absolute iteration to make resumed training and
// dispatch-size changes reproducible. SplitMix64 is ONLY the seed expander;
// the distributions below continue to use CUDA's multiply-with-carry RNG.
inline ulong BootstrapMixSeed(ulong value) {
    value += 0x9e3779b97f4a7c15ul;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ul;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebul;
    return value ^ (value >> 31);
}

inline ulong BootstrapSeedForItem(uint item, constant BootstrapParams& p) {
    const ulong base = (ulong(p.seed_high) << 32) | ulong(p.seed_low);
    // Distinct domains prevent swapping the item and iteration from creating
    // the same stream, or equal item/iteration hashes cancelling each other.
    ulong seed = BootstrapMixSeed(base
        ^ BootstrapMixSeed(ulong(p.iteration) ^ 0xd1b54a32d192ed03ul)
        ^ BootstrapMixSeed(((ulong(p.stream) << 32) | ulong(item)) ^ 0x94d049bb133111ebul));
    // All-zero halves are absorbing states for the corresponding MWC stream.
    if (uint(seed) == 0) seed |= 1ul;
    if (uint(seed >> 32) == 0) seed |= (1ul << 32);
    return seed;
}

inline uint BootstrapNextUint(thread ulong& seed) {
    uint v = uint(seed >> 32), u = uint(seed);
    v = 36969u * (v & 0xffffu) + (v >> 16);
    u = 18000u * (u & 0xffffu) + (u >> 16);
    seed = (ulong(v) << 32) | ulong(u);
    return (v << 16) + u;
}

inline float BootstrapUniform(thread ulong& seed) {
    // CUDA's double NextUniform is unavailable in MSL. Round to float as its
    // NextUniformF does, but clamp the upper endpoint to keep a proper [0,1)
    // variate after float32 rounding. PositiveUniform also excludes log(0).
    return min(float(BootstrapNextUint(seed)) * 2.3283064365386963e-10f,
               0x1.fffffep-1f);
}

inline float BootstrapPositiveUniform(thread ulong& seed) {
    return max(BootstrapUniform(seed), 0x1p-32f);
}

inline float BootstrapNextNormal(thread ulong& seed) {
    const float a = BootstrapPositiveUniform(seed);
    const float b = BootstrapUniform(seed);
    return sqrt(-2.0f * log(a)) * cos(6.2831853071795864769f * b);
}

inline float BootstrapNormalForItem(uint item, constant BootstrapParams& p) {
    ulong seed = BootstrapSeedForItem(item, p);
    for (uint step = 0; step < 4; ++step) BootstrapNextUint(seed);
    return BootstrapNextNormal(seed);
}

inline float BootstrapNextPoisson(thread ulong& seed, float lambda) {
    // random_gen.cuh::NextPoisson. Valid float32 subsample < 1 implies
    // lambda <= -log(2^-24), so bootstrap never reaches CUDA's lambda>20
    // normal approximation. The loop preserves the Poisson integer output.
    float log_probability = 0.0f;
    uint count = 0;
    do {
        ++count;
        log_probability += log(BootstrapPositiveUniform(seed));
    } while (log_probability > -lambda);
    return float(count - 1);
}

// Types: No=0, Bayesian=1, Bernoulli=2, Poisson=3, MVS=4. MVS has a
// separate threshold pass below. Runtime validation rejects unknown types,
// nonfinite parameters, negative temperature, and Poisson subsample >= 1.
kernel void GenerateBootstrapWeights(device float* multipliers [[buffer(0)]],
                                     const device float* derivatives [[buffer(1)]],
                                     constant BootstrapParams& p [[buffer(2)]],
                                     uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    ulong seed = BootstrapSeedForItem(row, p);
    float multiplier = 1.0f;
    if (p.type == 1 && p.temperature != 0.0f) {
        const float exponential = -log(BootstrapUniform(seed) + 1e-20f);
        multiplier = p.temperature == 1.0f ? exponential : pow(exponential, p.temperature);
    } else if (p.type == 2) {
        multiplier = float(BootstrapUniform(seed) < p.subsample);
    } else if (p.type == 3) {
        // -log(1-p) with a small-p correction when 1-p rounds to exactly 1.
        const float remainder = 1.0f - p.subsample;
        const float lambda = remainder == 1.0f ? p.subsample
            : -log(remainder) * (p.subsample / (1.0f - remainder));
        multiplier = BootstrapNextPoisson(seed, lambda);
    }
    multipliers[row] = multiplier;
}

// Keep the original observation weights untouched: CUDA bootstraps the weak
// target for structure search, then estimates final leaves on all objects.
kernel void ApplyBootstrapWeights(device float* gradients [[buffer(0)]],
                                  const device float* sample_weights [[buffer(1)]],
                                  const device float* multipliers [[buffer(2)]],
                                  device float* structure_weights [[buffer(3)]],
                                  constant BootstrapParams& p [[buffer(4)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const float multiplier = multipliers[row];
    // A excluded object contributes exactly zero, even when its derivative
    // is nonfinite. Runtime checks still reject nonfinite original inputs.
    gradients[row] = multiplier == 0.0f ? 0.0f : gradients[row] * multiplier;
    structure_weights[row] = multiplier == 0.0f ? 0.0f : sample_weights[row] * multiplier;
}

inline float BootstrapMvsMagnitude(float derivative, float lambda) {
    // sqrt(g*g+lambda) with scaling to prevent intermediate overflow.
    const float a = abs(derivative), b = sqrt(lambda);
    const float upper = max(a, b), lower = min(a, b);
    return upper == 0.0f ? 0.0f : upper * sqrt(1.0f + (lower / upper) * (lower / upper));
}

// First-tree automatic MVS regularization is square(mean(abs(g))). Reduce
// weighted objective gradients before sampling into at most a few thousand
// partial means, then sum .x in host double and square. The optional .y is
// the unweighted mean(g*g), NOT the weighted score-noise variance formula.
// Dispatch any positive number of full 256-thread groups; partials has one
// float2 per group. Normalize before accumulation to avoid sum overflow.
kernel void ReduceBootstrapStatistics(const device float* derivatives [[buffer(0)]],
                                      device float2* partials [[buffer(1)]],
                                      constant BootstrapParams& p [[buffer(2)]],
                                      uint tid [[thread_position_in_threadgroup]],
                                      uint group [[threadgroup_position_in_grid]],
                                      uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 scratch[256];
    float2 sum = float2(0.0f), correction = float2(0.0f);
    const float divisor = float(p.rows), root_divisor = sqrt(divisor);
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float gradient = derivatives[row];
        const float scaled = gradient / root_divisor;
        const float2 value = float2(abs(gradient) / divisor, scaled * scaled) - correction;
        const float2 next = sum + value;
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

// CUDA mvs.cu uses an 8192-object radix-sort/prefix-sum tile. This Metal
// adaptation retains the same tile boundary and inclusion-probability rule,
// but solves sum(min(1, magnitude/threshold)) = tile_size * subsample by
// monotone bisection. It needs 1 KiB of threadgroup scratch instead of CUB.
// Exactly 256 threads/group; output thresholds[ceil(rows / 8192)].
kernel void ComputeMvsThresholds(const device float* derivatives [[buffer(0)]],
                                 device float* thresholds [[buffer(1)]],
                                 constant BootstrapParams& p [[buffer(2)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tile [[threadgroup_position_in_grid]]) {
    threadgroup float scratch[256];
    const uint begin = tile * 8192;
    const uint count = min(8192u, p.rows - begin);
    float magnitudes[32];
    float local_max = 0.0f;
    for (uint i = 0; i < 32; ++i) {
        const uint index = i * 256 + tid;
        magnitudes[i] = index < count ? BootstrapMvsMagnitude(derivatives[begin + index], p.mvs_lambda) : 0.0f;
        local_max = max(local_max, magnitudes[i]);
    }
    scratch[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] = max(scratch[tid], scratch[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float scale = scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p.subsample >= 1.0f || scale == 0.0f) {
        if (tid == 0) thresholds[tile] = 0.0f;
        return;
    }
    // Normalize by the tile maximum so both the reduction and its threshold
    // interval remain representable for very large finite gradients.
    float local_sum = 0.0f;
    for (uint i = 0; i < 32; ++i) { magnitudes[i] /= scale; local_sum += magnitudes[i]; }
    scratch[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float sample_count = float(count) * p.subsample;
    float low = 0.0f, high = scratch[0] / sample_count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < 32; ++step) {
        const float threshold = low + (high - low) * 0.5f;
        float expected = 0.0f;
        for (uint i = 0; i < 32; ++i) expected += min(1.0f, magnitudes[i] / max(threshold, 1e-38f));
        scratch[tid] = expected;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 128; stride; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (scratch[0] > sample_count) low = threshold; else high = threshold;
        // Every thread must finish reading scratch[0] before the next step.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) thresholds[tile] = high * scale;
}

kernel void GenerateMvsBootstrapWeights(device float* multipliers [[buffer(0)]],
                                        const device float* derivatives [[buffer(1)]],
                                        const device float* thresholds [[buffer(2)]],
                                        constant BootstrapParams& p [[buffer(3)]],
                                        uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    if (p.subsample >= 1.0f) { multipliers[row] = 1.0f; return; }
    const float magnitude = BootstrapMvsMagnitude(derivatives[row], p.mvs_lambda);
    const float threshold = thresholds[row / 8192];
    const float probability = magnitude > threshold ? 1.0f
        : threshold > 0.0f ? magnitude / threshold : 0.0f;
    ulong seed = BootstrapSeedForItem(row, p);
    multipliers[row] = probability > 1.1920928955078125e-7f && BootstrapUniform(seed) < probability
        ? 1.0f / probability : 0.0f;
}

// Scalar score-noise primitive. CUDA Cosine applies one shared normal draw
// per FEATURE, not per threshold candidate; pass feature IDs to this helper.
kernel void GenerateBootstrapNormals(device float* normals [[buffer(0)]],
                                     constant BootstrapParams& p [[buffer(1)]],
                                     uint item [[thread_position_in_grid]]) {
    if (item < p.rows) normals[item] = BootstrapNormalForItem(item, p) * p.noise_scale;
}
)METAL";
