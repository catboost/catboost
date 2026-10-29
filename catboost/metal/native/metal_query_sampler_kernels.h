#pragma once

// CUDA gpu_data/querywise_helper.cpp and kernel/query_helper.cu. Input shuffle
// priorities and query masks are supplied separately from their RNG schedule.
static const char* CBMMetalQuerySamplerSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct QuerySamplerParams {
    uint rows, groups, max_query, pair_limit;
    float fraction; uint reserved0, reserved1, reserved2;
};

kernel void InitializeQuerySamplerRows(device uint* indices [[buffer(0)]],
    constant QuerySamplerParams& p [[buffer(1)]], uint row [[thread_position_in_grid]]) {
    if (row < p.rows) indices[row] = row;
}
kernel void GatherQuerySamplerKeys(const device uint* qids [[buffer(0)]],
    const device uint* indices [[buffer(1)]], device uint* keys [[buffer(2)]],
    constant QuerySamplerParams& p [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row < p.rows) keys[row] = qids[indices[row]];
}
kernel void CountQuerySamplerRows(const device uint* offsets [[buffer(0)]],
    const device float* query_mask [[buffer(1)]], device uint* counts [[buffer(2)]],
    device uint* live [[buffer(3)]], device uint* pairs [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]], constant QuerySamplerParams& p [[buffer(6)]],
    uint query [[thread_position_in_grid]]) {
    if (query >= p.groups) return;
    const float mask = query_mask[query];
    if (mask != 0.0f && mask != 1.0f) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    const uint size = offsets[query + 1] - offsets[query];
    const uint taken = mask > 0.0f ? min(p.max_query, max(min(2u, size), uint(ceil(p.fraction * float(size))))) : 0;
    counts[query] = taken; live[query] = uint(taken != 0);
    pairs[query] = min(taken * (taken - uint(taken > 0)) / 2, p.pair_limit + 1);
}
kernel void MarkQuerySamplerRows(const device uint* shuffled [[buffer(0)]],
    const device uint* qids [[buffer(1)]], const device uint* offsets [[buffer(2)]],
    const device uint* counts [[buffer(3)]], device uint* mask [[buffer(4)]],
    constant QuerySamplerParams& p [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint original = shuffled[row], query = qids[original];
    mask[original] = uint(row - offsets[query] < counts[query]);
}

// Saturating positive prefix addition avoids uint32 wrap on dense pair counts.
// Row/small-pair scans can use native SIMD sums. Larger pair-ID scans saturate
// each shuffle addition before it can overflow, including recursive totals.
kernel void ScanQuerySamplerPrefix(device uint* values [[buffer(0)]],
    device uint* totals [[buffer(1)]], constant QuerySamplerParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]], uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint parts[32];
    const uint index = block * 256 + tid, limit = p.pair_limit + 1;
    const uint value = index < p.rows ? min(values[index], limit) : 0;
    uint prefix = value;
    if (p.pair_limit <= (1u << 24)) {
        prefix = min(simd_prefix_inclusive_sum(value), limit);
    } else {
        for (uint distance = 1; distance < simd_width; distance <<= 1) {
            const uint previous = simd_shuffle_up(prefix, distance);
            if (lane >= distance) prefix += min(previous, limit - prefix);
        }
    }
    if (lane + 1 == simd_width) parts[simd_group] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = 0; group < simd_group; ++group) prefix += min(parts[group], limit - prefix);
    if (index < p.rows) values[index] = prefix;
    if (tid == 255) totals[block] = prefix;
}
kernel void AddQuerySamplerPrefix(device uint* values [[buffer(0)]],
    const device uint* parent [[buffer(1)]], constant QuerySamplerParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < p.rows && index >= 256)
        values[index] += min(parent[index / 256 - 1], p.pair_limit + 1 - values[index]);
}
kernel void ScatterQuerySamplerRows(const device uint* mask [[buffer(0)]],
    const device uint* row_prefix [[buffer(1)]], const device uint* qids [[buffer(2)]],
    const device uint* live_prefix [[buffer(3)]], device uint* documents [[buffer(4)]],
    device uint* sampled_qids [[buffer(5)]], constant QuerySamplerParams& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row < p.rows && mask[row]) {
        const uint index = row_prefix[row] - 1;
        documents[index] = row; sampled_qids[index] = live_prefix[qids[row]] - 1;
    }
}
kernel void ScatterQuerySamplerGroups(const device uint* counts [[buffer(0)]],
    const device uint* doc_prefix [[buffer(1)]], const device uint* live_prefix [[buffer(2)]],
    const device uint* pair_prefix [[buffer(3)]], device uint* offsets [[buffer(4)]],
    device uint* pair_offsets [[buffer(5)]], constant QuerySamplerParams& p [[buffer(6)]],
    uint query [[thread_position_in_grid]]) {
    if (query < p.groups && counts[query]) {
        const uint index = live_prefix[query] - 1;
        offsets[index] = query ? doc_prefix[query - 1] : 0;
        pair_offsets[index] = query ? pair_prefix[query - 1] : 0;
    }
}
kernel void FinalizeQuerySamplerShape(const device uint* row_prefix [[buffer(0)]],
    const device uint* doc_prefix [[buffer(1)]], const device uint* live_prefix [[buffer(2)]],
    const device uint* pair_prefix [[buffer(3)]], device uint* offsets [[buffer(4)]],
    device uint* pair_offsets [[buffer(5)]], device uint4* shape [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]], constant QuerySamplerParams& p [[buffer(8)]],
    uint index [[thread_position_in_grid]]) {
    if (index) return;
    const uint rows = row_prefix[p.rows - 1], groups = live_prefix[p.groups - 1], pairs = pair_prefix[p.groups - 1];
    if (rows != doc_prefix[p.groups - 1]) atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    if (pairs > p.pair_limit) atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
    offsets[groups] = rows; pair_offsets[groups] = pairs;
    shape[0] = uint4(rows, groups, pairs, 0);
}
)METAL";
