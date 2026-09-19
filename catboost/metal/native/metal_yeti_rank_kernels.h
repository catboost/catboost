#pragma once

// CUDA targets/kernel/yeti_rank_pointwise.cu. This is the pointwise YetiRank
// target, whose second output is incident pair weight, not logistic curvature.
static const char* CBMMetalYetiRankSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct YetiRankParams {
    uint rows, groups, tasks, permutations;
    uint seed_low, seed_high;
    float decay;
    uint center_rows;
};

inline uint YetiAdvance(thread uint& seed) {
    seed = 1664525u * seed + 1013904223u;
    return seed;
}

// center_rows is explicit because the pinned CUDA caller passes query count
// to RemoveQueryMeans' row-count argument. Passing rows instead repairs that
// upstream caller defect; it must not be silently confused with exact replay.
kernel void PrepareYetiRankApprox(const device float* point [[buffer(0)]],
    const device uint* offsets [[buffer(1)]], device float2* exponents [[buffer(2)]],
    constant YetiRankParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    const uint start = offsets[query], end = offsets[query + 1];
    float value = 0.0f;
    for (uint row = start + tid; row < end; row += 256) value += point[row] / float(end - start);
    partials[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) partials[tid] += partials[tid + width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row = start + tid; row < end; row += 256) {
        const float log_value = min(point[row] - (row < p.center_rows ? partials[0] : 0.0f), 70.0f);
        exponents[row] = float2(log_value, exp(log_value));
    }
}

inline bool YetiBefore(uint left, uint right, threadgroup const uint* qids,
    threadgroup const float* keys) {
    if (qids[left] != qids[right]) return qids[left] < qids[right];
    if (keys[left] != keys[right]) return keys[left] > keys[right];
    return left < right;
}

// Each task packs complete queries into at most1024 rows. The task's first
// global query ID and 256 lanes preserve CUDA's seed assignment and four
// random draws per lane, including padding. Stable bitonic tuple sorting
// replaces CUDA's descending float radix pass then stable query-ID pass.
// Total threadgroup storage is28KiB; there are no floating point atomics.
kernel void YetiRankPointwise(const device float2* exponents [[buffer(0)]],
    const device float* relevance [[buffer(1)]], const device float* weights [[buffer(2)]],
    const device uint* query_ids [[buffer(3)]], const device uint* offsets [[buffer(4)]],
    const device uint2* tasks [[buffer(5)]], device float2* derivatives [[buffer(6)]],
    constant YetiRankParams& p [[buffer(7)]], uint tid [[thread_position_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]]) {
    const uint first_query = tasks[task].x, end_query = tasks[task].y;
    const uint start = offsets[first_query], count = offsets[end_query] - start;
    threadgroup float approx[1024], relev[1024], keys[1024];
    threadgroup uint qids[1024], indices[1024];
    threadgroup float2 totals[1024];
    threadgroup atomic_uint log_mode;
    if (tid == 0) atomic_store_explicit(&log_mode, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < 4; ++k) {
        const uint index = tid + 256 * k;
        approx[index] = index < count ? exponents[start + index].y : 1000.0f;
        if (index < count && approx[index] < 1e-15f)
            atomic_store_explicit(&log_mode, 1u, memory_order_relaxed);
        relev[index] = index < count ? relevance[start + index] * weights[start + index] : 1000.0f;
        qids[index] = index < count ? query_ids[start + index] : end_query;
        totals[index] = float2(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool use_log = atomic_load_explicit(&log_mode, memory_order_relaxed) != 0u;
    uint seed = 127u * first_query + 16807u * tid + 1u;
    for (uint i = 0; i < 3; ++i) YetiAdvance(seed);
    seed += p.seed_low + p.seed_high;
    for (uint i = 0; i < 3; ++i) YetiAdvance(seed);
    for (uint permutation = 0; permutation < p.permutations; ++permutation) {
        for (uint k = 0; k < 4; ++k) {
            const uint index = tid + 256 * k;
            const float uniform = float(YetiAdvance(seed)) * 2.328306435996595e-10f;
            keys[index] = use_log
                ? (index < count ? exponents[start + index].x + log(uniform) - log(1.000001f - uniform) : -INFINITY)
                : (index < count ? approx[index] : -1000.0f) * (uniform / (1.000001f - uniform));
            indices[index] = index;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint length = 2; length <= 1024; length <<= 1) {
            for (uint stride = length >> 1; stride; stride >>= 1) {
                uint output[4];
                for (uint k = 0; k < 4; ++k) {
                    const uint slot = tid + 256 * k, partner = slot ^ stride;
                    const uint a = indices[slot], b = indices[partner];
                    const bool take_first = ((slot & length) == 0) == ((slot & stride) == 0);
                    output[k] = YetiBefore(a, b, qids, keys) == take_first ? a : b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < 4; ++k) indices[tid + 256 * k] = output[k];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // Preserve the source's k-major previous/current accumulation order.
        for (uint k = 0; k < 4; ++k) {
            const uint slot = tid + 256 * k;
            const bool paired = slot < count && slot && qids[slot] == qids[slot - 1];
            const uint right = indices[slot], left = paired ? indices[slot - 1] : right;
            float gradient = 0.0f, mass = 0.0f;
            if (paired) {
                const uint rank = slot - (offsets[qids[slot]] - start);
                mass = (0.15f * pow(p.decay, float(rank - 1))) * abs(relev[left] - relev[right]) / float(p.permutations);
                const float denominator = approx[left] + approx[right];
                if (!use_log) {
                    gradient = mass * (relev[left] > relev[right] ? approx[right] : -approx[left]) / denominator;
                } else {
                    // Equivalent log-domain evaluation prevents 0/0 and lost
                    // permutation ordering when both exponentials underflow.
                    const float difference = exponents[start + left].x - exponents[start + right].x;
                    const float small = exp(-abs(difference));
                    const float right_probability = difference >= 0.0f ? small / (1.0f + small) : 1.0f / (1.0f + small);
                    const float left_probability = difference >= 0.0f ? 1.0f / (1.0f + small) : small / (1.0f + small);
                    gradient = mass * (relev[left] > relev[right] ? right_probability : -left_probability);
                }
                totals[left] += float2(gradient, mass);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (paired) totals[right] += float2(-gradient, mass);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint k = 0; k < 4; ++k) {
        const uint index = tid + 256 * k;
        if (index < count) derivatives[start + index] = totals[index];
    }
}
)METAL";
