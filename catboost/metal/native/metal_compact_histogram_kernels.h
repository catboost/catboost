#pragma once

// Self-contained MSL source; may be appended to the other training sources.
// The host mirrors CompactHistogramParams (32 bytes) and validates:
//   0 < rows <= 2^24; leaves is a power of two <= 65536;
//   reuse is 0 or 1, and reuse implies leaves >= 2;
//   features > 0; feature_begin+features fits the input feature matrix;
//   tile_rows is a positive multiple of 256;
//   feature_offsets[0] == 0, feature_offsets[features] == total_bins > 0;
//   every feature span is in [0,256], and every candidate bin is in its span.
// Features without candidates may have zero spans. A span is max candidate
// bin+1, NOT necessarily the number of observed data bins: terminal numeric
// overflow and unknown one-hot values outside the span are ignored.
//
// Histogram cell = leaf*total_bins + feature_offsets[local_feature] + bin.
// All cell addresses use 64-bit arithmetic. Two statistic buffers cost
// 8*leaves*total_bins bytes; enforce checked host size arithmetic, the device
// buffer limit and the total workspace budget before allocating/dispatching.
//
// Work buffers:
//   state: 4 uints {job_count, active_count, error, reserved}, atomically used;
//   jobs: job_capacity uint4s {row_begin,row_end,storage_leaf,data_leaf};
//   active: (reuse ? leaves/2 : leaves) uint leaf/parent ids;
//   indirect_args: 9 packed uints, three 12-byte MTL dispatch argument records.
// A safe job_capacity is ceil(rows/tile_rows)+min(rows,leaves). Builders guard
// every job write and set error=1 on insufficient capacity. The host must
// report a nonzero state[2]; dependent dispatches suppress work on that error.
//
// Order: reset state; build jobs; build indirect args; clear histograms;
// compute (indirect offset 0); scan (offset 12); for reuse subtract (offset 24).
// Use separate ordered compute encoders or equivalent buffer barriers.
// Compute/scan/subtract use exactly 256 threads/group. Before reuse, preserve
// scanned parents in the left half, supply NEW partitions, and keep gradients,
// weights and feature layout unchanged. Empty parent slots must contain zero.
//
// feature_begin selects a contiguous feature tile in the original matrix;
// feature_offsets and histogram cells are local to that tile. If retaining all
// feature histograms exceeds the budget, stream tiles with reuse=0 at every
// depth, combine split winners using original candidate indices for ties,
// and discard the tile. Reuse requires retaining every tile's parent cache.
// Partition construction and split scoring are separate runtime contracts.
static const char* CBMMetalCompactHistogramSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

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

struct CompactHistogramParams {
    uint rows, features, leaves, total_bins;
    uint feature_begin, tile_rows, job_capacity, reuse;
};

// Threads: 1. Reset independently of histogram clearing so full and reuse
// builds have the same metadata lifecycle.
kernel void ResetCompactHistogramWorkState(device atomic_uint* state [[buffer(0)]],
                                           constant CompactHistogramParams& p [[buffer(1)]],
                                           uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        for (uint i = 0; i < 4; ++i)
            atomic_store_explicit(state + i, 0u, memory_order_relaxed);
    }
}

// Threads: reuse ? leaves/2 : leaves. Compact nonempty partition work before
// feature dispatch. CUDA's ComputeSplitPropertiesPass partitions row work;
// this explicit list avoids leaf*feature dispatches for empty Metal leaves.
// Smaller-child selection is TPointwisePartOffsetsHelper's strict size test.
// Active reuse entries include nonempty parents even when the selected child
// is empty, since subtraction still has to restore the other child.
kernel void BuildCompactHistogramJobs(
    const device uint* partition_offsets [[buffer(0)]],
    device uint4* jobs [[buffer(1)]],
    device uint* active [[buffer(2)]],
    device atomic_uint* state [[buffer(3)]],
    constant CompactHistogramParams& p [[buffer(4)]],
    uint leaf [[thread_position_in_grid]]) {
    const uint partitions = p.reuse ? p.leaves / 2 : p.leaves;
    if (leaf >= partitions) return;
    uint selected = leaf;
    uint storage = leaf;
    uint size = partition_offsets[leaf + 1] - partition_offsets[leaf];
    if (p.reuse) {
        const uint right = leaf + partitions;
        const uint right_size = partition_offsets[right + 1] - partition_offsets[right];
        if (size + right_size == 0) return;
        selected = size < right_size ? leaf : right;
        size = min(size, right_size);
        storage = right;
    } else if (size == 0) {
        return;
    }
    const uint active_index = atomic_fetch_add_explicit(state + 1, 1u, memory_order_relaxed);
    active[active_index] = leaf;
    if (size == 0) return;
    const uint count = 1 + (size - 1) / p.tile_rows;
    const uint first = atomic_fetch_add_explicit(state, count, memory_order_relaxed);
    if (first > p.job_capacity || count > p.job_capacity - first) {
        atomic_store_explicit(state + 2, 1u, memory_order_relaxed);
        return;
    }
    const uint begin = partition_offsets[selected];
    const uint end = partition_offsets[selected + 1];
    for (uint tile = 0; tile < count; ++tile) {
        const uint start = begin + tile * p.tile_rows;
        jobs[first + tile] = uint4(start, min(start + p.tile_rows, end), storage, selected);
    }
}

// Threads: 1. Output uint[9], not uint3[3] (MSL uint3 has 16-byte alignment).
// Byte offsets 0/12/24 describe compute/scan/subtract threadgroup counts.
// Zero group counts are intentional for empty work or metadata failure.
kernel void BuildCompactHistogramDispatchArguments(
    const device atomic_uint* state [[buffer(0)]],
    device uint* arguments [[buffer(1)]],
    constant CompactHistogramParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const bool valid = atomic_load_explicit(state + 2, memory_order_relaxed) == 0;
    const uint jobs = valid ? atomic_load_explicit(state, memory_order_relaxed) : 0;
    const uint active = valid ? atomic_load_explicit(state + 1, memory_order_relaxed) : 0;
    arguments[0] = jobs;
    arguments[1] = p.features;
    arguments[2] = 1;
    arguments[3] = active;
    arguments[4] = p.features;
    arguments[5] = 1;
    const ulong cells = p.reuse ? ulong(active) * p.total_bins : 0;
    arguments[6] = uint((cells + 255) / 256);
    arguments[7] = 1;
    arguments[8] = 1;
}

// Threads: (reuse ? leaves/2 : leaves)*total_bins. Full builds clear all
// storage; reuse clears only right scratch slots, preserving cached parents.
kernel void ClearCompactHistograms(device float* sums [[buffer(0)]],
                                   device float* weights [[buffer(1)]],
                                   constant CompactHistogramParams& p [[buffer(2)]],
                                   uint tid [[thread_position_in_grid]]) {
    const ulong cells = ulong(p.reuse ? p.leaves / 2 : p.leaves) * p.total_bins;
    if (ulong(tid) < cells) {
        const ulong cell = (p.reuse ? cells : 0) + tid;
        sums[cell] = 0.0f;
        weights[cell] = 0.0f;
    }
}

// Indirect offset 0: (job_count,features,1), 256 threads, 16 KiB local memory.
// CUDA TCFeature::{FirstFoldIndex,Folds} supplies the compact feature layout;
// ComputeSplitPropertiesPass supplies block-local accumulation/global merge.
// p.feature_begin translates tile-local features to original data columns.
kernel void ComputeCompactHistograms(
    const device uchar* bins [[buffer(0)]],
    const device float* derivatives [[buffer(1)]],
    const device float* sample_weights [[buffer(2)]],
    const device uint* row_indices [[buffer(3)]],
    const device uint* feature_offsets [[buffer(4)]],
    const device uint4* jobs [[buffer(5)]],
    const device atomic_uint* state [[buffer(6)]],
    device atomic_float* sums [[buffer(7)]],
    device atomic_float* weights [[buffer(8)]],
    constant CompactHistogramParams& p [[buffer(9)]],
    uint3 local [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (atomic_load_explicit(state + 2, memory_order_relaxed) != 0 ||
        group.x >= atomic_load_explicit(state, memory_order_relaxed) ||
        group.x >= p.job_capacity || group.y >= p.features) return;
    const uint first_bin = feature_offsets[group.y];
    const uint bin_count = feature_offsets[group.y + 1] - first_bin;
    if (bin_count == 0) return;
    const uint tid = local.x;
    const uint4 job = jobs[group.x];
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
    for (uint position = job.x + tid; position < job.y; position += 256) {
        const uint row = row_indices[position];
        const uint bin = uint(bins[(ulong(p.feature_begin) + group.y) * p.rows + row]);
        // Out-of-span bins do not belong to any candidate histogram. In
        // particular, never fold a numeric overflow bin into the last border.
        if (bin < bin_count) {
            HistogramAtomicAdd(local_sums + bank + bin, sum_errors + bank + bin, derivatives[row]);
            HistogramAtomicAdd(local_weights + bank + bin, weight_errors + bank + bin, sample_weights[row]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < bin_count) {
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
        const ulong cell = ulong(job.z) * p.total_bins + first_bin + tid;
        if (sum.x != 0.0f || sum.y != 0.0f)
            atomic_fetch_add_explicit(sums + cell, sum.x + sum.y, memory_order_relaxed);
        if (weight.x != 0.0f || weight.y != 0.0f)
            atomic_fetch_add_explicit(weights + cell, weight.x + weight.y, memory_order_relaxed);
    }
}

// Indirect offset 12: (active_count,features,1), 256 threads, 4 KiB local.
// ScanHistogramsImpl prefixes numeric candidates; one-hot equality bins stay
// raw. Reuse scans the right storage half irrespective of selected data side.
kernel void ScanCompactHistograms(
    device float* sums [[buffer(0)]],
    device float* weights [[buffer(1)]],
    const device uchar* feature_types [[buffer(2)]],
    const device uint* feature_offsets [[buffer(3)]],
    const device uint* active [[buffer(4)]],
    const device atomic_uint* state [[buffer(5)]],
    constant CompactHistogramParams& p [[buffer(6)]],
    uint3 local [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    if (atomic_load_explicit(state + 2, memory_order_relaxed) != 0 ||
        group.x >= atomic_load_explicit(state + 1, memory_order_relaxed) ||
        group.y >= p.features || feature_types[p.feature_begin + group.y] != 0) return;
    const uint first_bin = feature_offsets[group.y];
    const uint bin_count = feature_offsets[group.y + 1] - first_bin;
    if (bin_count == 0) return;
    const uint tid = local.x;
    const uint leaf = active[group.x] + (p.reuse ? p.leaves / 2 : 0);
    const ulong cell = ulong(leaf) * p.total_bins + first_bin + tid;
    threadgroup float2 local_sums[256];
    threadgroup float2 local_weights[256];
    local_sums[tid] = float2(tid < bin_count ? sums[cell] : 0.0f, 0.0f);
    local_weights[tid] = float2(tid < bin_count ? weights[cell] : 0.0f, 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const float2 previous_sum = tid >= stride ? local_sums[tid - stride] : 0.0f;
        const float2 previous_weight = tid >= stride ? local_weights[tid - stride] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        local_sums[tid] = HistogramMergePair(local_sums[tid], previous_sum);
        local_weights[tid] = HistogramMergePair(local_weights[tid], previous_weight);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < bin_count) {
        sums[cell] = local_sums[tid].x + local_sums[tid].y;
        weights[cell] = local_weights[tid].x + local_weights[tid].y;
    }
}

// Indirect offset 24: (ceil(active_count*total_bins/256),1,1), 256 threads.
// UpdatePointwiseHistogramsImpl restores actual child slots with unclamped
// parent-minus-child statistics. Sparse active parents avoid empty-pair work.
kernel void SubtractCompactSiblingHistograms(
    device float* sums [[buffer(0)]],
    device float* weights [[buffer(1)]],
    const device uint* partition_offsets [[buffer(2)]],
    const device uint* active [[buffer(3)]],
    const device atomic_uint* state [[buffer(4)]],
    constant CompactHistogramParams& p [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    if (!p.reuse || atomic_load_explicit(state + 2, memory_order_relaxed) != 0) return;
    const uint active_count = atomic_load_explicit(state + 1, memory_order_relaxed);
    if (ulong(tid) >= ulong(active_count) * p.total_bins) return;
    const uint left = active[tid / p.total_bins];
    const uint right = left + p.leaves / 2;
    const uint left_size = partition_offsets[left + 1] - partition_offsets[left];
    const uint right_size = partition_offsets[right + 1] - partition_offsets[right];
    const bool left_calculated = left_size < right_size;
    const uint bin = tid % p.total_bins;
    const ulong left_cell = ulong(left) * p.total_bins + bin;
    const ulong right_cell = ulong(right) * p.total_bins + bin;
    const float calculated_sum = sums[right_cell];
    const float calculated_weight = weights[right_cell];
    const float complement_sum = sums[left_cell] - calculated_sum;
    const float complement_weight = weights[left_cell] - calculated_weight;
    sums[left_cell] = left_calculated ? calculated_sum : complement_sum;
    weights[left_cell] = left_calculated ? calculated_weight : complement_weight;
    sums[right_cell] = left_calculated ? complement_sum : calculated_sum;
    weights[right_cell] = left_calculated ? complement_weight : calculated_weight;
}
)METAL";
