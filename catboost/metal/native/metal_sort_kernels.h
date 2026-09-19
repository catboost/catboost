#pragma once

// Standalone source, also safe to concatenate with other Metal kernel headers.
// Stable 4-bit LSD radix passes use exactly 256 threads per group. Supported
// SIMD execution widths divide 256 and are >= 8 (Apple Silicon uses 32).
// CBMSortKernelParams = four uint32 values {rows, tiles, shift, elements}.
//
// One radix pass:
// 1. SortCountTiles(keys, row_rank, digit_major_hist, params), tiles groups.
// 2. Recursively SortScanPrefixes(prefix, block_totals, params),
//    ceil(elements/256) groups, starting elements=tiles*16. Each level scans
//    in place within its groups and writes totals to the next level.
// 3. Descending levels: SortAddBlockPrefixes(child, parent, params),
//    ceil(child_elements/256) groups. Top-level inclusive prefix is already
//    global; each child adds the preceding parent block's inclusive sum.
// 4. SortScatter(keys, payload, row_rank, scanned_hist, out_keys, out_payload,
//    params), tiles groups. Flattened digit-major prefixes incorporate both
//    all smaller digits and all preceding tiles of the current digit.
//
// Eight passes with shifts 0,4,...,28 produce stable full uint32 ordering.
static const char* CBMSortMetalSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct CBMSortKernelParams {
    uint rows, tiles, shift, elements;
};

kernel void SortInitializePayload(device uint* payload [[buffer(0)]],
                                  constant CBMSortKernelParams& p [[buffer(1)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row < p.rows) payload[row] = row;
}

kernel void SortCountTiles(const device uint* keys [[buffer(0)]],
                           device uint* row_rank [[buffer(1)]],
                           device uint* histogram [[buffer(2)]],
                           constant CBMSortKernelParams& p [[buffer(3)]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tile [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint simd_width [[threads_per_simdgroup]]) {
    // One independent SIMD prefix per radix digit. A short scan of preceding
    // SIMD counts then yields each row's stable rank in its 256-row tile.
    threadgroup uint simd_counts[16 * 32];
    const uint row = tile * 256 + tid;
    const bool active = row < p.rows;
    const uint digit = active ? ((keys[row] >> p.shift) & 15u) : 0;
    uint rank = 0;
    for (uint bin = 0; bin < 16; ++bin) {
        const uint selected = uint(active && digit == bin);
        const uint prefix = simd_prefix_inclusive_sum(selected);
        if (selected) rank = prefix - 1;
        if (lane + 1 == simd_width) simd_counts[simd_group * 16 + bin] = prefix;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        for (uint group = 0; group < simd_group; ++group) rank += simd_counts[group * 16 + digit];
        row_rank[row] = rank;
    }
    if (tid < 16) {
        uint total = 0;
        for (uint group = 0; group < 256 / simd_width; ++group) total += simd_counts[group * 16 + tid];
        histogram[tid * p.tiles + tile] = total;
    }
}

kernel void SortScanPrefixes(device uint* prefix_values [[buffer(0)]],
                             device uint* block_totals [[buffer(1)]],
                             constant CBMSortKernelParams& p [[buffer(2)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint block [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd_group [[simdgroup_index_in_threadgroup]],
                             uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint simd_totals[32];
    const uint cell = block * 256 + tid;
    const uint value = cell < p.elements ? prefix_values[cell] : 0;
    uint prefix = simd_prefix_inclusive_sum(value);
    if (lane + 1 == simd_width) simd_totals[simd_group] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = 0; group < simd_group; ++group) prefix += simd_totals[group];
    if (cell < p.elements) prefix_values[cell] = prefix;
    if (tid == 255) block_totals[block] = prefix;
}

kernel void SortAddBlockPrefixes(device uint* prefixes [[buffer(0)]],
                                 const device uint* parent_prefixes [[buffer(1)]],
                                 constant CBMSortKernelParams& p [[buffer(2)]],
                                 uint cell [[thread_position_in_grid]]) {
    if (cell < p.elements && cell >= 256) prefixes[cell] += parent_prefixes[cell / 256 - 1];
}

kernel void SortScatter(const device uint* keys [[buffer(0)]],
                        const device uint* payload [[buffer(1)]],
                        const device uint* row_rank [[buffer(2)]],
                        const device uint* histogram [[buffer(3)]],
                        device uint* out_keys [[buffer(4)]],
                        device uint* out_payload [[buffer(5)]],
                        constant CBMSortKernelParams& p [[buffer(6)]],
                        uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint key = keys[row];
    const uint digit = (key >> p.shift) & 15u;
    const uint cell = digit * p.tiles + row / 256;
    const uint destination = (cell ? histogram[cell - 1] : 0) + row_rank[row];
    out_keys[destination] = key;
    out_payload[destination] = payload[row];
}
)METAL";
