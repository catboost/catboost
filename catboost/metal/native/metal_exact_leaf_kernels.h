#pragma once

#include <cstdint>

// Host/MSL ABI. Alpha is 0.5 for MAE/MAPE. Exact leaf estimation uses neither
// Quantile delta nor leaf regularization, iteration count or backtracking.
struct CBMExactLeafParams {
    uint32_t Rows, Leaves, IsMape, TilesPerLeaf;
    float Alpha;
    uint32_t Reserved0, Reserved1, Reserved2;
};
static_assert(sizeof(CBMExactLeafParams) == 32, "Exact leaf parameter ABI mismatch");

// Append after CBMMetalObjectiveSource. Compile with fastMathEnabled = NO.
//
// Encode in this order, using distinct input/output radix-sort buffers:
//  1. PrepareExactResiduals: rows threads.
//  2. Stable-sort (ordered_keys,row_ids) by all 32 key bits.
//  3. MakeExactLeafKeys: rows threads, retaining the sorted row-ID payload.
//  4. Stable-sort (leaf_keys,sorted_row_ids) by all 32 key bits.
//  5. BuildExactLeafOffsets: leaves+1 threads.
//  6. ReduceExactLeafPartials: (tiles_per_leaf,leaves) groups of 256 threads.
//  7. PrefixExactLeafTiles: leaves groups of 256 threads.
//  8. SelectExactLeafQuantile: (tiles_per_leaf,leaves) groups of 256 threads.
//  9. FinalizeExactLeafValues: leaves threads.
//
// Scratch: residuals/effective_weights float[rows]; sort keys/payload uint[rows]
// plus sort-owned scratch; offsets uint[leaves+1]; partials float4[leaves*tiles];
// tile_offsets float2[leaves*tiles]; totals float2[leaves]; selected uint[leaves].
// Outputs: unshrunk raw_values float[leaves], ORIGINAL output_weights float[leaves].
// Require rows/leaves > 0, leaf_ids < leaves, tiles_per_leaf in [1,256], finite
// alpha in [0,1]. Inputs and scratch stay on the GPU throughout this sequence.
static const char* CBMMetalExactLeafSource = R"METAL(

struct ExactLeafParams {
    uint rows, leaves, is_mape, tiles_per_leaf;
    float alpha;
    uint reserved0, reserved1, reserved2;
};

inline float2 ExactLeafAdd(float2 left, float2 right) {
    float high = left.x, low = left.y;
    ObjectiveAddExpansion(high, low, right.x);
    ObjectiveAddExpansion(high, low, right.y);
    return float2(high, low);
}

inline float2 ExactLeafMultiply(float2 value, float multiplier) {
    const float high = value.x * multiplier;
    const float low = fma(value.x, multiplier, -high);
    return ExactLeafAdd(float2(high, low), float2(value.y * multiplier, 0.0f));
}

inline bool ExactLeafAtLeast(float2 prefix, float2 target) {
    const float2 difference = ExactLeafAdd(prefix, -target);
    return difference.x > 0.0f || (difference.x == 0.0f && difference.y >= 0.0f);
}

inline uint2 ExactLeafTileRange(uint begin, uint end, uint tile,
                                constant ExactLeafParams& p) {
    const uint count = end - begin;
    const uint span = count / p.tiles_per_leaf + uint(count % p.tiles_per_leaf != 0);
    const ulong start = min(ulong(tile) * span, ulong(count));
    const ulong stop = min(ulong(tile + 1u) * span, ulong(count));
    return uint2(begin + uint(start), begin + uint(stop));
}

// CUDA DerCalcer::ComputeExactValue produces float residuals. For MAPE, retain
// the loss's ORIGINAL target denominator when adjusting sample weights: using
// the residual as CUDA does can increase MAPE after an Exact update. The full
// ordered-float key also fixes CUDA's lossy segmented radix bit range.
kernel void PrepareExactResiduals(
    const device float* targets [[buffer(0)]],
    const device float* sample_weights [[buffer(1)]],
    const device float* predictions [[buffer(2)]],
    device float* residuals [[buffer(3)]],
    device float* effective_weights [[buffer(4)]],
    device uint* ordered_keys [[buffer(5)]],
    device uint* row_ids [[buffer(6)]],
    constant ExactLeafParams& p [[buffer(7)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    float residual = targets[row] - predictions[row];
    // Both signed zeros compare equal; normalizing their keys retains row-order
    // tie breaking instead of imposing a separate negative-zero bucket.
    if (residual == 0.0f) residual = 0.0f;
    const float weight = sample_weights[row];
    const bool valid = isfinite(residual) && isfinite(weight) && weight >= 0.0f;
    residuals[row] = residual;
    effective_weights[row] = valid
        ? (p.is_mape ? weight / max(1.0f, abs(targets[row])) : weight)
        : ObjectiveInvalidValue();
    const uint bits = as_type<uint>(residual);
    ordered_keys[row] = bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
    row_ids[row] = row;
}

kernel void MakeExactLeafKeys(
    const device uint* sorted_rows [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]],
    device uint* leaf_keys [[buffer(2)]],
    constant ExactLeafParams& p [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
    if (position < p.rows) leaf_keys[position] = leaf_ids[sorted_rows[position]];
}

// Lower bounds handle any number of rows and every empty leaf, including the
// first and last. No fixed-iteration binary search or empty-segment reads.
kernel void BuildExactLeafOffsets(
    const device uint* sorted_leaf_keys [[buffer(0)]],
    device uint* offsets [[buffer(1)]],
    constant ExactLeafParams& p [[buffer(2)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf > p.leaves) return;
    uint begin = 0, end = p.rows;
    while (begin < end) {
        const uint middle = begin + (end - begin) / 2;
        if (sorted_leaf_keys[middle] < leaf) begin = middle + 1;
        else end = middle;
    }
    offsets[leaf] = begin;
}

kernel void ReduceExactLeafPartials(
    const device uint* sorted_rows [[buffer(0)]],
    const device float* effective_weights [[buffer(1)]],
    const device float* sample_weights [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device float4* partials [[buffer(4)]],
    constant ExactLeafParams& p [[buffer(5)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    threadgroup float4 high_scratch[256], low_scratch[256];
    const uint tile = group.x, leaf = group.y;
    const uint2 range = ExactLeafTileRange(offsets[leaf], offsets[leaf + 1], tile, p);
    float3 high = float3(0.0f), low = float3(0.0f);
    for (ulong position = ulong(range.x) + tid; position < range.y; position += 256) {
        const uint row = sorted_rows[position];
        ObjectiveAddExpansion(high, low, float3(effective_weights[row], 0.0f,
                                                sample_weights[row]));
    }
    high_scratch[tid] = float4(high, 0.0f);
    low_scratch[tid] = float4(low, 0.0f);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) partials[leaf * p.tiles_per_leaf + tile] = float4(
        high_scratch[0].x, low_scratch[0].x,
        high_scratch[0].z, low_scratch[0].z);
}

kernel void PrefixExactLeafTiles(
    const device float4* partials [[buffer(0)]],
    device float2* tile_offsets [[buffer(1)]],
    device float2* totals [[buffer(2)]],
    device float* output_weights [[buffer(3)]],
    device atomic_uint* selected [[buffer(4)]],
    constant ExactLeafParams& p [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float4 scratch[256];
    const uint cell = leaf * p.tiles_per_leaf + tid;
    scratch[tid] = tid < p.tiles_per_leaf ? partials[cell] : float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const float4 left = tid >= stride ? scratch[tid - stride] : float4(0.0f);
        float4 current = scratch[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= stride) current = float4(ExactLeafAdd(left.xy, current.xy),
                                            ExactLeafAdd(left.zw, current.zw));
        scratch[tid] = current;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.tiles_per_leaf) tile_offsets[cell] = tid ? scratch[tid - 1].xy : float2(0.0f);
    if (tid == 0) {
        const float4 total = scratch[p.tiles_per_leaf - 1];
        totals[leaf] = total.xy;
        output_weights[leaf] = total.z + total.w;
        atomic_store_explicit(selected + leaf, p.alpha == 1.0f ? 0u : 0xffffffffu,
                              memory_order_relaxed);
    }
}

kernel void SelectExactLeafQuantile(
    const device uint* sorted_rows [[buffer(0)]],
    const device float* effective_weights [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device float2* tile_offsets [[buffer(3)]],
    const device float2* totals [[buffer(4)]],
    device atomic_uint* selected [[buffer(5)]],
    constant ExactLeafParams& p [[buffer(6)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x;
    threadgroup float2 scratch[256];
    const uint tile = group.x, leaf = group.y;
    const float2 total = totals[leaf];
    if (!all(isfinite(total)) || !(total.x + total.y > 0.0f)) return;
    const uint begin = offsets[leaf], end = offsets[leaf + 1];
    const uint2 range = ExactLeafTileRange(begin, end, tile, p);
    if (p.alpha == 0.0f) {
        if (tile == 0 && tid == 0 && begin < end)
            atomic_store_explicit(selected + leaf, begin, memory_order_relaxed);
        return;
    }
    if (p.alpha == 1.0f) {
        for (ulong position = ulong(range.x) + tid; position < range.y; position += 256) {
            if (effective_weights[sorted_rows[position]] > 0.0f)
                atomic_fetch_max_explicit(selected + leaf, uint(position) + 1u,
                                          memory_order_relaxed);
        }
        return;
    }
    // Use the weighted lower bound itself. CUDA's absolute FLT_EPSILON shift
    // changes the selected quantile when every sample weight is scaled down.
    // Keep compensated arithmetic through comparison, without that shift.
    const float2 target = ExactLeafMultiply(total, p.alpha);
    float2 previous = tile_offsets[leaf * p.tiles_per_leaf + tile];
    uint first = 0xffffffffu;
    for (ulong chunk = range.x; chunk < range.y; chunk += 256) {
        const ulong position = chunk + tid;
        const float weight = position < range.y
            ? effective_weights[sorted_rows[position]] : 0.0f;
        scratch[tid] = float2(weight, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 256; stride <<= 1) {
            const float2 left = tid >= stride ? scratch[tid - stride] : float2(0.0f);
            float2 current = scratch[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid >= stride) current = ExactLeafAdd(left, current);
            scratch[tid] = current;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float2 prefix = ExactLeafAdd(previous, scratch[tid]);
        if (position < range.y && ExactLeafAtLeast(prefix, target))
            first = min(first, uint(position));
        previous = ExactLeafAdd(previous, scratch[255]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (first != 0xffffffffu)
        atomic_fetch_min_explicit(selected + leaf, first, memory_order_relaxed);
}

kernel void FinalizeExactLeafValues(
    const device uint* sorted_rows [[buffer(0)]],
    const device float* residuals [[buffer(1)]],
    const device uint* offsets [[buffer(2)]],
    const device float2* totals [[buffer(3)]],
    const device atomic_uint* selected [[buffer(4)]],
    device float* raw_values [[buffer(5)]],
    constant ExactLeafParams& p [[buffer(6)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const float2 total = totals[leaf];
    if (!all(isfinite(total))) {
        raw_values[leaf] = ObjectiveInvalidValue();
        return;
    }
    if (offsets[leaf] == offsets[leaf + 1] || !(total.x + total.y > 0.0f)) {
        raw_values[leaf] = 0.0f;
        return;
    }
    uint position = atomic_load_explicit(selected + leaf, memory_order_relaxed);
    if (p.alpha == 1.0f) position = position ? position - 1u : 0xffffffffu;
    raw_values[leaf] = position >= offsets[leaf] && position < offsets[leaf + 1]
        ? residuals[sorted_rows[position]] : ObjectiveInvalidValue();
}
)METAL";
