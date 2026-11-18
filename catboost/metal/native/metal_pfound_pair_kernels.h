#pragma once

// CUDA targets/kernel/pfound_f.cu: PFoundF/YetiRankPairwise's stochastic
// non-diagonal target on an already-sampled, ordered query view.
static const char* CBMMetalPFoundPairSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct PFoundPairParams {
    uint rows, groups, tasks, pairs;
    uint permutations, seed_low, seed_high, reserved;
    float decay; uint reserved1, reserved2, reserved3;
};
inline uint PFoundAdvance(thread uint& seed) {
    seed = 1664525u * seed + 1013904223u;
    return seed;
}
inline uint PFoundPairIndex(uint a, uint b) {
    const uint lo = min(a, b), hi = max(a, b);
    return hi * (hi - 1) / 2 + lo;
}
inline bool PFoundBefore(uint a, uint b, threadgroup const uint* qids, threadgroup const float* keys) {
    if (qids[a] != qids[b]) return qids[a] < qids[b];
    if (keys[a] != keys[b]) return keys[a] > keys[b];
    return a < b;
}

kernel void PreparePFoundPairApprox(const device float* point [[buffer(0)]],
    const device uint* offsets [[buffer(1)]], device float* exponents [[buffer(2)]],
    constant PFoundPairParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    threadgroup float parts[256];
    const uint begin = offsets[query], end = offsets[query + 1];
    float value = -INFINITY;
    for (uint row = begin + tid; row < end; row += 256) value = max(value, point[row]);
    parts[tid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) parts[tid] = max(parts[tid], parts[tid + width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint row = begin + tid; row < end; row += 256)
        exponents[row] = exp(point[row] - parts[0]);
}

// 1024-row task packing and four random draws per 256 lanes match CUDA.
// Query IDs stay uint: CUDA's uchar queryId aliases queries above 255 in
// tasks with many short queries. Each real adjacent pair has one writer per
// permutation; device barriers serialize repeated contributions without atomics.
kernel void GeneratePFoundPairWeights(const device float* exponents [[buffer(0)]],
    const device float* relevance [[buffer(1)]], const device uint* query_ids [[buffer(2)]],
    const device uint* offsets [[buffer(3)]], const device uint2* tasks [[buffer(4)]],
    const device uint* pair_offsets [[buffer(5)]], device float* matrix [[buffer(6)]],
    constant PFoundPairParams& p [[buffer(7)]], uint tid [[thread_position_in_threadgroup]],
    uint task [[threadgroup_position_in_grid]]) {
    const uint first_query = tasks[task].x, end_query = tasks[task].y;
    const uint start = offsets[first_query], count = offsets[end_query] - start;
    threadgroup float approx[1024], relev[1024], keys[1024];
    threadgroup uint qids[1024], indices[1024];
    for (uint k = 0; k < 4; ++k) {
        const uint index = tid + 256 * k;
        approx[index] = index < count ? exponents[start + index] : 1000.0f;
        relev[index] = index < count ? relevance[start + index] : 1000.0f;
        qids[index] = index < count ? query_ids[start + index] : end_query;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cuda_seed = p.seed_low + p.seed_high;
    uint seed = 127u * first_query + 16807u * tid + cuda_seed * (1u + first_query);
    PFoundAdvance(seed); seed += cuda_seed;
    for (uint i = 0; i < 3; ++i) PFoundAdvance(seed);
    for (uint permutation = 0; permutation < p.permutations; ++permutation) {
        for (uint k = 0; k < 4; ++k) {
            const uint index = tid + 256 * k;
            const float uniform = float(PFoundAdvance(seed)) * 2.328306435996595e-10f;
            keys[index] = (index < count ? approx[index] : -1000.0f) * (uniform / (1.000001f - uniform));
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
                    output[k] = PFoundBefore(a, b, qids, keys) == take_first ? a : b;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint k = 0; k < 4; ++k) indices[tid + 256 * k] = output[k];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        for (uint k = 0; k < 4; ++k) {
            const uint slot = tid + 256 * k;
            if (slot < count && slot && qids[slot] == qids[slot - 1]) {
                const uint left = indices[slot - 1], right = indices[slot], query = qids[slot];
                const uint begin = offsets[query] - start, rank = slot - begin;
                const float weight = (0.15f * pow(p.decay, float(rank - 1))) * abs(relev[left] - relev[right]) / float(p.permutations);
                matrix[pair_offsets[query] + PFoundPairIndex(left - begin, right - begin)] += weight;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    }
}

// CUDA filters raw (optionally bootstrapped) matrix weights BEFORE multiplying
// by the lower-index sampled document's query weight. Its weak Hessian uses
// this mass for every score enum; it is not logistic curvature.
kernel void FinalizePFoundPairs(const device float* matrix [[buffer(0)]],
    const device float* multipliers [[buffer(1)]], const device float* relevance [[buffer(2)]],
    const device float* weights [[buffer(3)]], const device float* exponents [[buffer(4)]],
    const device uint* offsets [[buffer(5)]], const device uint* pair_offsets [[buffer(6)]],
    const device uint* document_ids [[buffer(7)]], device uint2* pairs [[buffer(8)]],
    device float4* edges [[buffer(9)]], device atomic_uint* status [[buffer(10)]],
    constant PFoundPairParams& p [[buffer(11)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.pairs) return;
    uint lo = 0, hi = p.groups;
    while (lo < hi) { const uint mid = (lo + hi + 1) / 2;
        if (pair_offsets[mid] <= index) lo = mid; else hi = mid - 1; }
    const uint local = index - pair_offsets[lo];
    uint b = uint((1.0f + sqrt(8.0f * float(local) + 1.0f)) * 0.5f);
    const uint a = local - b * (b - 1) / 2;
    const uint left = offsets[lo] + a, right = offsets[lo] + b;
    pairs[index] = uint2(document_ids[left], document_ids[right]);
    const float raw = matrix[index] * multipliers[index];
    const float mass = abs(raw) > 1e-20f ? raw * weights[left] : 0.0f;
    const float ax = exponents[left] + 1e-20f, ay = exponents[right] + 1e-20f;
    const float gradient = mass * (relevance[left] > relevance[right] ? ay : -ax) / (ax + ay);
    edges[index] = float4(gradient, mass, mass, 0.0f);
    if (!isfinite(raw) || !isfinite(mass) || !isfinite(gradient) || mass < 0.0f)
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

// Deterministic incident-gradient reduction replaces source floating atomics.
// Results remain in sampled-document order; pairs already contain original IDs.
kernel void ReducePFoundPairGradients(const device float4* edges [[buffer(0)]],
    const device uint* query_ids [[buffer(1)]], const device uint* offsets [[buffer(2)]],
    const device uint* pair_offsets [[buffer(3)]], device float2* gradient [[buffer(4)]],
    constant PFoundPairParams& p [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint query = query_ids[row], begin = offsets[query], local = row - begin;
    float sum = 0.0f, correction = 0.0f;
    for (uint other = 0; other < offsets[query + 1] - begin; ++other) {
        if (other == local) continue;
        const float value = edges[pair_offsets[query] + PFoundPairIndex(local, other)].x * (local < other ? 1.0f : -1.0f);
        const float next = sum + value;
        correction += abs(sum) >= abs(value) ? (sum - next) + value : (value - next) + sum;
        sum = next;
    }
    gradient[row] = float2(sum, correction);
}
)METAL";
