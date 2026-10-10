#pragma once

// Appended to the common Metal source after KernelParams and SplitState.
// All cooperative kernels require exactly 256 threads/group.
// p.tile_rows must be a positive multiple of 256, p.leaves/p.bins <= 256,
// p.reserved0 is the bounded histogram tile count, and p.reserved1 is
// ceil(p.rows / p.tile_rows). Buffers and dispatches are described below.
static const char* CBMMetalHistogramSource = R"METAL(

#ifndef CBM_HISTOGRAM_EXPANSIONS
#define CBM_HISTOGRAM_EXPANSIONS
// Keep rounding residuals through local histogram collisions and prefix
// reductions. Final device histogram storage remains float32.
inline float2 HistogramMergePair(float2 left, float2 right) {
    const float sum = left.x + right.x;
    const float part = sum - left.x;
    const float error = (left.x - (sum - part)) + (right.x - part);
    const float tail = (left.y + right.y) + error;
    const float high = sum + tail;
    const float high_part = high - sum;
    return float2(high, (sum - (high - high_part)) + (tail - high_part));
}
inline void HistogramAtomicAdd(threadgroup atomic_uint* high,
                                threadgroup atomic_uint* low, float value) {
    uint previous = atomic_load_explicit(high, memory_order_relaxed);
    float error;
    while (true) {
        const float old = as_type<float>(previous);
        const float sum = old + value;
        const float part = sum - old;
        error = (old - (sum - part)) + (value - part);
        if (atomic_compare_exchange_weak_explicit(high, &previous, as_type<uint>(sum),
                memory_order_relaxed, memory_order_relaxed)) break;
    }
    if (error != 0.0f) {
        previous = atomic_load_explicit(low, memory_order_relaxed);
        while (!atomic_compare_exchange_weak_explicit(low, &previous,
            as_type<uint>(as_type<float>(previous) + error),
            memory_order_relaxed, memory_order_relaxed)) {}
    }
}
#endif

// CUDA pointwise_optimization_subsets.cpp::TSubsetsHelper::Split sorts row
// indices by leaf before rebuilding partitions. This Metal foundation performs
// a stable counting scatter using tile-local ranks. Sorting fixed 256-row
// chunks makes ranks independent of atomic execution order. Workspace is
// rows * sizeof(uint) for ranks plus tiles * leaves * sizeof(uint) for counts;
// the latter is reused for exclusive tile offsets by PrefixPartitionTiles.
// Groups: (p.reserved1,1,1), 256 threads. Threadgroup memory: 4 KiB.
kernel void CountPartitionTiles(const device uint* leaf_ids [[buffer(0)]],
                                device uint* row_ranks [[buffer(1)]],
                                device uint* tile_offsets [[buffer(2)]],
                                constant KernelParams& p [[buffer(3)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tile [[threadgroup_position_in_grid]]) {
    const uint begin = tile * p.tile_rows;
    const uint end = min(begin + p.tile_rows, p.rows);
    if (p.leaves == 1) {
        for (uint row = begin + tid; row < end; row += 256) row_ranks[row] = row - begin;
        if (tid == 0) tile_offsets[tile] = end - begin;
        return;
    }
    threadgroup uint sorted_keys[256];
    threadgroup uint leaf_counts[256];
    leaf_counts[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint chunk = begin; chunk < end; chunk += 256) {
        const uint row = chunk + tid;
        // The low byte preserves input order among equal leaf keys. Padded
        // rows use a leaf sentinel strictly greater than every valid leaf.
        uint key = ((row < end ? leaf_ids[row] : 256u) << 8) | tid;
        sorted_keys[tid] = key;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint width = 2; width <= 256; width <<= 1) {
            for (uint stride = width >> 1; stride; stride >>= 1) {
                const uint other = sorted_keys[tid ^ stride];
                const bool take_min = ((tid & width) == 0) == ((tid & stride) == 0);
                key = take_min ? min(key, other) : max(key, other);
                // All neighbor reads must complete before any overwrite.
                threadgroup_barrier(mem_flags::mem_threadgroup);
                sorted_keys[tid] = key;
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        const uint leaf = key >> 8;
        uint first = 0;
        if (leaf < p.leaves) {
            // Lower bound of this leaf in the sorted fixed-size chunk.
            uint upper = tid;
            while (first < upper) {
                const uint middle = (first + upper) >> 1;
                if ((sorted_keys[middle] >> 8) < leaf) first = middle + 1;
                else upper = middle;
            }
            row_ranks[chunk + (key & 255u)] = leaf_counts[leaf] + tid - first;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // One writer per leaf, after all threads have read the prior count.
        if (leaf < p.leaves &&
            (tid == 255 || (sorted_keys[tid + 1] >> 8) != leaf)) {
            leaf_counts[leaf] += tid - first + 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.leaves) tile_offsets[tile * p.leaves + tid] = leaf_counts[tid];
}

// Counts become exclusive offsets within each leaf, then a threadgroup scan
// constructs partition_offsets[leaves+1]. One group, 256 threads, 1 KiB local.
kernel void PrefixPartitionTiles(device uint* tile_offsets [[buffer(0)]],
                                 device uint* partition_offsets [[buffer(1)]],
                                 constant KernelParams& p [[buffer(2)]],
                                 uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint leaf_prefix[256];
    uint count = 0;
    if (tid < p.leaves) {
        for (uint tile = 0; tile < p.reserved1; ++tile) {
            const uint cell = tile * p.leaves + tid;
            const uint tile_count = tile_offsets[cell];
            tile_offsets[cell] = count;
            count += tile_count;
        }
    }
    leaf_prefix[tid] = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const uint previous = tid >= stride ? leaf_prefix[tid - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        leaf_prefix[tid] += previous;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.leaves) partition_offsets[tid] = leaf_prefix[tid] - count;
    if (tid == 0) partition_offsets[p.leaves] = leaf_prefix[255];
}

// Threads: rows. Output indices are in ascending original-row order within
// each leaf; empty leaves have equal adjacent offsets. No floating atomics or
// nondeterministic global scatter counters participate in partitioning.
kernel void ScatterPartitionRows(const device uint* leaf_ids [[buffer(0)]],
                                  const device uint* row_ranks [[buffer(1)]],
                                  const device uint* tile_offsets [[buffer(2)]],
                                  const device uint* partition_offsets [[buffer(3)]],
                                  device uint* row_indices [[buffer(4)]],
                                  constant KernelParams& p [[buffer(5)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row < p.rows) {
        const uint leaf = leaf_ids[row];
        const uint tile = row / p.tile_rows;
        const uint destination = partition_offsets[leaf]
                               + tile_offsets[tile * p.leaves + leaf]
                               + row_ranks[row];
        row_indices[destination] = row;
    }
}

// Threads: leaves*features*bins. These are weighted floating statistics, not
// integer document counts or Newton Hessians. Clear before a full rebuild.
kernel void ClearHistograms(device float* sums [[buffer(0)]],
                            device float* weights [[buffer(1)]],
                            constant KernelParams& p [[buffer(2)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < p.leaves * p.features * p.bins) {
        sums[i] = 0;
        weights[i] = 0;
    }
}

// CUDA pointwise_hist2_one_byte_templ.cuh::ComputeSplitPropertiesPass builds
// local derivative/weight histograms before merging block totals globally.
// This Metal version uses four banks, eight SIMD groups and unpacked uint8
// features. It intentionally rebuilds each leaf; cached-parent subtraction
// requires a separate histogram-lifetime contract.
// Groups: (p.reserved0, p.features, p.leaves), 256 threads; local memory: 16 KiB.
// Each group scans one strided portion of a partition, then contributes at
// most bins*2 global additions regardless of its number of rows. Native
// device float atomics require Metal 3.0 / Apple GPU family 7 or newer.
kernel void ComputeHistograms(const device uchar* bins [[buffer(0)]],
                              const device float* derivatives [[buffer(1)]],
                              const device float* sample_weights [[buffer(2)]],
                              const device uint* row_indices [[buffer(3)]],
                              const device uint* partition_offsets [[buffer(4)]],
                              device atomic_float* sums [[buffer(5)]],
                              device atomic_float* weights [[buffer(6)]],
                              constant KernelParams& p [[buffer(7)]],
                              uint3 thread_position [[thread_position_in_threadgroup]],
                              uint3 group [[threadgroup_position_in_grid]]) {
    const uint tid = thread_position.x;
    const uint begin = partition_offsets[group.z];
    const uint end = partition_offsets[group.z + 1];
    if (begin + group.x * 256 >= end) return;
    threadgroup atomic_uint local_sums[1024];
    threadgroup atomic_uint local_weights[1024];
    threadgroup atomic_uint sum_errors[1024], weight_errors[1024];
    for (uint cell = tid; cell < 1024; cell += 256) {
        atomic_store_explicit(local_sums + cell, 0u, memory_order_relaxed);
        atomic_store_explicit(local_weights + cell, 0u, memory_order_relaxed);
        atomic_store_explicit(sum_errors + cell, 0u, memory_order_relaxed);
        atomic_store_explicit(weight_errors + cell, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint bank = ((tid >> 5) & 3u) * 256;
    for (uint position = begin + group.x * 256 + tid; position < end;
         position += p.reserved0 * 256) {
        const uint row = row_indices[position];
        const uint bin = uint(bins[ulong(group.y) * p.rows + row]);
        HistogramAtomicAdd(local_sums + bank + bin, sum_errors + bank + bin, derivatives[row]);
        HistogramAtomicAdd(local_weights + bank + bin, weight_errors + bank + bin, sample_weights[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < p.bins) {
        float2 sum = 0.0f, weight = 0.0f;
        for (uint bank_id = 0; bank_id < 4; ++bank_id) {
            const uint local_cell = bank_id * 256 + tid;
            sum = HistogramMergePair(sum, float2(
                as_type<float>(atomic_load_explicit(local_sums + local_cell, memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(sum_errors + local_cell, memory_order_relaxed))));
            weight = HistogramMergePair(weight, float2(
                as_type<float>(atomic_load_explicit(local_weights + local_cell, memory_order_relaxed)),
                as_type<float>(atomic_load_explicit(weight_errors + local_cell, memory_order_relaxed))));
        }
        const uint cell = (group.z * p.features + group.y) * p.bins + tid;
        if (sum.x != 0.0f || sum.y != 0.0f)
            atomic_fetch_add_explicit(sums + cell, sum.x + sum.y, memory_order_relaxed);
        if (weight.x != 0.0f || weight.y != 0.0f)
            atomic_fetch_add_explicit(weights + cell, weight.x + weight.y, memory_order_relaxed);
    }
}

// CUDA split_properties_helpers.cuh::ScanHistogramsImpl prefixes numeric
// bins and leaves one-hot histograms as exact equality bins. Inclusive
// parallel scans preserve these semantics; reduction order differs from CUDA.
// Groups: (leaves*features,1,1), 256 threads; threadgroup memory: 4 KiB.
kernel void ScanHistograms(device float* sums [[buffer(0)]],
                           device float* weights [[buffer(1)]],
                           const device uchar* feature_types [[buffer(2)]],
                           constant KernelParams& p [[buffer(3)]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint group [[threadgroup_position_in_grid]]) {
    if (feature_types[group % p.features] != 0) return;
    threadgroup float2 local_sums[256];
    threadgroup float2 local_weights[256];
    const uint cell = group * p.bins + tid;
    local_sums[tid] = float2(tid < p.bins ? sums[cell] : 0.0f, 0.0f);
    local_weights[tid] = float2(tid < p.bins ? weights[cell] : 0.0f, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const float2 previous_sum = tid >= stride ? local_sums[tid - stride] : 0.0f;
        const float2 previous_weight = tid >= stride ? local_weights[tid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_sums[tid] = HistogramMergePair(local_sums[tid], previous_sum);
        local_weights[tid] = HistogramMergePair(local_weights[tid], previous_weight);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.bins) {
        sums[cell] = local_sums[tid].x + local_sums[tid].y;
        weights[cell] = local_weights[tid].x + local_weights[tid].y;
    }
}

// CUDA models/kernel/add_model_value.cu::ComputeObliviousTreeBinsImpl:
// numeric splits set a bit for bin>border; one-hot splits set it for equality.
// The selected GPU state avoids host scalar readback before updating bins.
// Threads: rows. Rebuild partitions afterwards with the new p.leaves.
kernel void UpdateLeafBins(const device uchar* bins [[buffer(0)]],
                           device uint* leaf_ids [[buffer(1)]],
                           const device SplitState* selected [[buffer(2)]],
                           constant KernelParams& p [[buffer(3)]],
                           uint row [[thread_position_in_grid]]) {
    if (row < p.rows && selected->valid) {
        const uint value = uint(bins[ulong(selected->feature) * p.rows + row]);
        const uint bit = selected->type == 0 ? value > selected->bin
                                             : value == selected->bin;
        leaf_ids[row] |= bit << p.split_level;
    }
}
)METAL";
