#pragma once

// Append after the shared KernelParams definition. This source complements
// metal_histogram_kernels.h and reuses its PrefixPartitionTiles kernel.
static const char* CBMMetalIncrementalPartitionSource = R"METAL(

// CUDA TSubsetsHelper::CreateSubsets starts with identity document indices and
// one root partition. Threads: p.rows; row leaf IDs are initialized separately.
kernel void InitializeRootPartition(device uint* row_indices [[buffer(0)]],
                                     device uint* partition_offsets [[buffer(1)]],
                                     constant KernelParams& p [[buffer(2)]],
                                     uint row [[thread_position_in_grid]]) {
    if (row < p.rows) row_indices[row] = row;
    if (row == 0) {
        partition_offsets[0] = 0;
        partition_offsets[1] = p.rows;
    }
}

// CUDA TSubsetsHelper::Split reorders only the newly appended split bit. The
// input indices here are already grouped by the previous leaf, so previous-
// leaf segments are contiguous within each 256-position chunk. A segmented
// inclusive scan counts each side of the new bit in eight steps, retaining
// input order and avoiding a complete leaf-key sort at every tree depth.
//
// p.leaves is the NEW leaf count (a power of two, 2..256); leaf_ids already
// contain the new split bit. p.tile_rows is a positive multiple of 256 and
// p.reserved1 = ceil(p.rows / p.tile_rows). Groups: (p.reserved1,1,1), exactly
// 256 threads. Threadgroup memory: 4 KiB. row_ranks has p.rows uint entries,
// indexed by position in old_row_indices; tile_offsets has tiles*leaves uint
// entries. PrefixPartitionTiles changes the latter counts into tile offsets.
kernel void CountIncrementalPartitionTiles(
    const device uint* old_row_indices [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]],
    device uint* row_ranks [[buffer(2)]],
    device uint* tile_offsets [[buffer(3)]],
    constant KernelParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
    threadgroup uint previous_leaves[256];
    threadgroup uint left_prefix[256];
    threadgroup uint right_prefix[256];
    threadgroup uint leaf_counts[256];
    leaf_counts[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint previous_leaf_count = p.leaves >> 1;
    const uint begin = tile * p.tile_rows;
    const uint end = min(begin + p.tile_rows, p.rows);
    for (uint chunk = begin; chunk < end; chunk += 256) {
        const uint position = chunk + tid;
        const bool valid = position < end;
        const uint leaf = valid ? leaf_ids[old_row_indices[position]] : 256u;
        const uint previous_leaf = valid ? leaf & (previous_leaf_count - 1) : 256u;
        const bool right = (leaf & previous_leaf_count) != 0;
        previous_leaves[tid] = previous_leaf;
        left_prefix[tid] = uint(valid && !right);
        right_prefix[tid] = uint(valid && right);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = 1; stride < 256; stride <<= 1) {
            const bool same_segment = tid >= stride
                                   && previous_leaves[tid - stride] == previous_leaf;
            const uint left_add = same_segment ? left_prefix[tid - stride] : 0;
            const uint right_add = same_segment ? right_prefix[tid - stride] : 0;
            // Complete neighbor reads before changing the shared prefixes.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            left_prefix[tid] += left_add;
            right_prefix[tid] += right_add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (valid) {
            const uint rank = right ? right_prefix[tid] : left_prefix[tid];
            row_ranks[position] = leaf_counts[leaf] + rank - 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Each previous-leaf segment has one last thread. It updates both
        // child counts after every thread has read their prior chunk totals.
        if (valid && (tid == 255 || previous_leaves[tid + 1] != previous_leaf)) {
            leaf_counts[previous_leaf] += left_prefix[tid];
            leaf_counts[previous_leaf + previous_leaf_count] += right_prefix[tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.leaves) tile_offsets[tile * p.leaves + tid] = leaf_counts[tid];
}

// Run existing PrefixPartitionTiles between count and scatter with p.leaves
// still equal to the NEW leaf count. All buffers except output indices may
// be reused between depths. old_row_indices and new_row_indices MUST be
// distinct buffers; swap their host handles after encoding this dispatch.
// Threads: p.rows. Stable previous-leaf ordering followed by stable new-bit
// partitioning preserves original row order inside each resulting leaf.
kernel void ScatterIncrementalPartitionRows(
    const device uint* old_row_indices [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]],
    const device uint* row_ranks [[buffer(2)]],
    const device uint* tile_offsets [[buffer(3)]],
    const device uint* partition_offsets [[buffer(4)]],
    device uint* new_row_indices [[buffer(5)]],
    constant KernelParams& p [[buffer(6)]],
    uint position [[thread_position_in_grid]]) {
    if (position < p.rows) {
        const uint row = old_row_indices[position];
        const uint leaf = leaf_ids[row];
        const uint tile = position / p.tile_rows;
        const uint destination = partition_offsets[leaf]
                               + tile_offsets[tile * p.leaves + leaf]
                               + row_ranks[position];
        new_row_indices[destination] = row;
    }
}
)METAL";
