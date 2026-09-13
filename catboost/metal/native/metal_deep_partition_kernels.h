#pragma once

// Append after the shared KernelParams definition. These kernels implement a
// stable radix partition on the newly appended leaf bit, as in CUDA's
// TSubsetsHelper::Split / ReorderBins(..., currentDepth, 1). Existing row indices
// must already be ordered by the previous leaf and leaf_ids must contain the
// new split bit. The previous leaf ordering is preserved within each bit half.
//
// p.rows must be positive and p.leaves is the NEW power-of-two leaf count,
// 2..65536 for symmetric trees through depth 16. No reserved params are used.
// Exactly 256 threads/group are required by the three scan kernels.
//
// Workspace (all uint): row_prefix[rows], tile_prefix[ceil(rows/256)],
// block_prefix[ceil(rows/65536)], plus distinct input/output row index arrays
// and distinct input/output partition offsets. Prefix workspace is about
// 4.016 bytes/row and has no multiplicative dependence on the leaf count.
// The existing InitializeRootPartition establishes the first input buffers.
static const char* CBMMetalDeepPartitionSource = R"METAL(

// SIMD prefix plus a short sum of preceding SIMD totals. Two barriers make
// the shared scratch reusable in the next scan. The allocation supports any
// execution width dividing 256, without assuming a CUDA warp size.
inline uint DeepPartitionScan256(uint value, threadgroup uint* simd_totals,
                                 uint simd_lane, uint simd_group, uint simd_width) {
    uint prefix = simd_prefix_inclusive_sum(value);
    if (simd_lane + 1 == simd_width) simd_totals[simd_group] = prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint preceding = 0; preceding < simd_group; ++preceding) {
        prefix += simd_totals[preceding];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return prefix;
}

// Step 1. Groups: ceil(rows/256), 256 threads; threadgroup memory: 1 KiB.
// Each row receives the inclusive right-bit count within its 256-row tile.
// tile_prefix initially contains one total per tile; no buffer clear needed.
kernel void CountDeepPartitionBits(
    const device uint* old_row_indices [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]],
    device uint* row_prefix [[buffer(2)]],
    device uint* tile_prefix [[buffer(3)]],
    constant KernelParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint scratch[256];
    const uint position = tile * 256 + tid;
    const uint right = position < p.rows
        ? uint((leaf_ids[old_row_indices[position]] & (p.leaves >> 1)) != 0) : 0;
    const uint prefix = DeepPartitionScan256(right, scratch, simd_lane, simd_group, simd_width);
    if (position < p.rows) row_prefix[position] = prefix;
    if (tid == 255) tile_prefix[tile] = prefix;
}

// Step 2. Groups: ceil(rows/65536), 256 threads; local memory: 1 KiB.
// Scan tile totals in place within blocks of 256 tiles and emit block totals.
kernel void ScanDeepPartitionTiles(
    device uint* tile_prefix [[buffer(0)]],
    device uint* block_prefix [[buffer(1)]],
    constant KernelParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint scratch[256];
    const uint tiles = (p.rows - 1) / 256 + 1;
    const uint tile = block * 256 + tid;
    const uint count = tile < tiles ? tile_prefix[tile] : 0;
    const uint prefix = DeepPartitionScan256(count, scratch, simd_lane, simd_group, simd_width);
    if (tile < tiles) tile_prefix[tile] = prefix;
    if (tid == 255) block_prefix[block] = prefix;
}

// Step 3. One group, 256 threads; local memory: 1028 bytes. With the current
// 2^24-row runtime limit all block totals fit one scan. The chunk loop keeps
// this prefix kernel valid if that row limit is raised in a future runtime.
kernel void ScanDeepPartitionBlocks(
    device uint* block_prefix [[buffer(0)]],
    constant KernelParams& p [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint simd_width [[threads_per_simdgroup]]) {
    threadgroup uint scratch[256];
    threadgroup uint next_carry;
    const uint blocks = (p.rows - 1) / 65536 + 1;
    uint carry = 0;
    for (uint begin = 0; begin < blocks; begin += 256) {
        const uint block = begin + tid;
        const uint count = block < blocks ? block_prefix[block] : 0;
        const uint prefix = DeepPartitionScan256(count, scratch, simd_lane, simd_group, simd_width);
        if (block < blocks) block_prefix[block] = carry + prefix;
        if (tid == 255) next_carry = carry + prefix;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        carry = next_carry;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Reconstruct a global inclusive right count without a separate propagation
// pass. Prefix buffers respectively contain row-within-tile, tile-within-block
// and global block counts.
inline uint DeepPartitionRightThrough(
    uint position, const device uint* row_prefix,
    const device uint* tile_prefix, const device uint* block_prefix) {
    const uint tile = position / 256;
    const uint block = tile / 256;
    uint right = row_prefix[position];
    if ((tile & 255u) != 0) right += tile_prefix[tile - 1];
    if (block != 0) right += block_prefix[block - 1];
    return right;
}

// Step 4. Threads: previous leaf count (p.leaves/2). A stable global bit
// partition places all bit-zero rows before all bit-one rows. For a parent
// starting at s, its children start at s-rightBefore(s) and
// totalLeft+rightBefore(s). Repeated offsets for empty parents remain valid,
// including parents starting at zero or at rows. old_offsets and new_offsets
// MUST be distinct buffers because child offsets overlap unread parent slots.
kernel void BuildDeepPartitionOffsets(
    const device uint* old_offsets [[buffer(0)]],
    const device uint* row_prefix [[buffer(1)]],
    const device uint* tile_prefix [[buffer(2)]],
    const device uint* block_prefix [[buffer(3)]],
    device uint* new_offsets [[buffer(4)]],
    constant KernelParams& p [[buffer(5)]],
    uint parent [[thread_position_in_grid]]) {
    const uint parents = p.leaves >> 1;
    if (parent < parents) {
        const uint blocks = (p.rows - 1) / 65536 + 1;
        const uint total_left = p.rows - block_prefix[blocks - 1];
        const uint start = old_offsets[parent];
        const uint right_before = start != 0
            ? DeepPartitionRightThrough(start - 1, row_prefix, tile_prefix, block_prefix) : 0;
        new_offsets[parent] = start - right_before;
        new_offsets[parent + parents] = total_left + right_before;
        if (parent == 0) new_offsets[p.leaves] = p.rows;
    }
}

// Step 5. Threads: rows. Distinct row index buffers are required. A right row
// has destination totalLeft+rightThrough(position)-1; a left row has
// destination position-rightThrough(position). Every output position has one
// writer and original row order within each new leaf remains stable. Swap
// both row-index and offset buffer handles after encoding these dispatches.
kernel void ScatterDeepPartitionRows(
    const device uint* old_row_indices [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]],
    const device uint* row_prefix [[buffer(2)]],
    const device uint* tile_prefix [[buffer(3)]],
    const device uint* block_prefix [[buffer(4)]],
    device uint* new_row_indices [[buffer(5)]],
    constant KernelParams& p [[buffer(6)]],
    uint position [[thread_position_in_grid]]) {
    if (position < p.rows) {
        const uint row = old_row_indices[position];
        const bool right = (leaf_ids[row] & (p.leaves >> 1)) != 0;
        const uint prefix = DeepPartitionRightThrough(position, row_prefix, tile_prefix, block_prefix);
        const uint blocks = (p.rows - 1) / 65536 + 1;
        const uint total_left = p.rows - block_prefix[blocks - 1];
        const uint destination = right ? total_left + prefix - 1 : position - prefix;
        new_row_indices[destination] = row;
    }
}
)METAL";
