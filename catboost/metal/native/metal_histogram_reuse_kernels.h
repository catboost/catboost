#pragma once

// Append after CBMMetalSource, CBMMetalObjectiveSource and
// CBMMetalHistogramSource, which supply KernelParams and HistogramAtomicAdd.
// The caller retains the previous level's scanned parent histograms in the
// first half of the buffers and rebuilds row partitions for the NEW leaves.
// All four dispatches below use that new p.leaves (an even power of two,
// 2..256), p.bins <= 256, and an unchanged feature/statistic layout. Gradients
// and structure weights must be unchanged since the parent was calculated.
// Dispatch in order: clear, compute, scan, subtract. Each cooperative kernel
// requires exactly 256 threads/group; p.reserved0 is a positive tile count.
static const char* CBMMetalHistogramReuseSource = R"METAL(

// Threads: (p.leaves/2)*p.features*p.bins. Preserve the cached parents in the
// left storage half; the right half will hold whichever child is smaller.
kernel void ClearChildHistograms(device float* sums [[buffer(0)]],
                                 device float* weights [[buffer(1)]],
                                 constant KernelParams& p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
    const uint parent_cells = (p.leaves / 2) * p.features * p.bins;
    if (i < parent_cells) {
        sums[parent_cells + i] = 0.0f;
        weights[parent_cells + i] = 0.0f;
    }
}

// CUDA TPointwisePartOffsetsHelper::ShiftPartAndBinSumsPtr chooses the left
// child only when it is strictly smaller, choosing right for equal sizes.
// Its histogram is always accumulated into the right STORAGE slot, even
// when the selected DATA partition is the left child. Native device float
// atomics and the four-bank threadgroup CAS scheme match ComputeHistograms.
// Groups: (p.reserved0,p.features,p.leaves/2), 256 threads, 16 KiB local memory.
// partition_offsets has p.leaves+1 entries for the newly split row_indices.
kernel void ComputeSmallerChildHistograms(
    const device uchar* bins [[buffer(0)]],
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
    const uint left = group.z;
    const uint right = left + p.leaves / 2;
    const uint left_size = partition_offsets[left + 1] - partition_offsets[left];
    const uint right_size = partition_offsets[right + 1] - partition_offsets[right];
    const uint selected = left_size < right_size ? left : right;
    const uint begin = partition_offsets[selected];
    const uint end = partition_offsets[selected + 1];
    // This condition is uniform for the entire group, including empty leaves.
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
        const uint cell = (right * p.features + group.y) * p.bins + tid;
        if (sum.x != 0.0f || sum.y != 0.0f)
            atomic_fetch_add_explicit(sums + cell, sum.x + sum.y, memory_order_relaxed);
        if (weight.x != 0.0f || weight.y != 0.0f)
            atomic_fetch_add_explicit(weights + cell, weight.x + weight.y, memory_order_relaxed);
    }
}

// CUDA ScanPointwiseHistograms offsets an incremental scan to the right
// storage half. Numeric features receive inclusive border prefixes; one-hot
// features retain raw equality-bin statistics. Cached parents are untouched.
// Groups: ((p.leaves/2)*p.features,1,1), 256 threads, 4 KiB local memory.
kernel void ScanChildHistograms(device float* sums [[buffer(0)]],
                                device float* weights [[buffer(1)]],
                                const device uchar* feature_types [[buffer(2)]],
                                constant KernelParams& p [[buffer(3)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]]) {
    if (feature_types[group % p.features] != 0) return;
    threadgroup float2 local_sums[256];
    threadgroup float2 local_weights[256];
    const uint storage_group = (p.leaves / 2) * p.features + group;
    const uint cell = storage_group * p.bins + tid;
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

// CUDA pointwise_hist2.cu::UpdatePointwiseHistogramsImpl computes the other
// child by direct float subtraction and restores actual left/right ordering.
// No weight clamp: CUDA preserves the subtraction result, including roundoff.
// Threads: (p.leaves/2)*p.features*p.bins. Each thread owns both sibling cells.
kernel void SubtractSiblingHistograms(device float* sums [[buffer(0)]],
                                      device float* weights [[buffer(1)]],
                                      const device uint* partition_offsets [[buffer(2)]],
                                      constant KernelParams& p [[buffer(3)]],
                                      uint i [[thread_position_in_grid]]) {
    const uint parent_count = p.leaves / 2;
    const uint cells_per_leaf = p.features * p.bins;
    const uint parent_cells = parent_count * cells_per_leaf;
    if (i >= parent_cells) return;
    const uint left = i / cells_per_leaf;
    const uint right = left + parent_count;
    const uint left_size = partition_offsets[left + 1] - partition_offsets[left];
    const uint right_size = partition_offsets[right + 1] - partition_offsets[right];
    const bool left_calculated = left_size < right_size;
    const uint right_cell = parent_cells + i;
    const float calculated_sum = sums[right_cell];
    const float calculated_weight = weights[right_cell];
    const float complement_sum = sums[i] - calculated_sum;
    const float complement_weight = weights[i] - calculated_weight;
    sums[i] = left_calculated ? calculated_sum : complement_sum;
    weights[i] = left_calculated ? calculated_weight : complement_weight;
    sums[right_cell] = left_calculated ? complement_sum : calculated_sum;
    weights[right_cell] = left_calculated ? complement_weight : calculated_weight;
}
)METAL";
