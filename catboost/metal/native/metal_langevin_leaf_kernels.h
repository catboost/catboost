#pragma once

// Additive scalar leaf direction; append after shared backtracking source.
static const char* CBMMetalLangevinLeafSource = R"METAL(
kernel void PrepareLangevinBacktrackingDirection(
    const device float4* partials [[buffer(0)]],
    const device float* current_values [[buffer(1)]],
    device float* directions [[buffer(2)]],
    device float* output_weights [[buffer(3)]],
    device float2* direction_dot [[buffer(4)]],
    const device float2* gradient_noise [[buffer(5)]],
    const device float2* diagonal_noise [[buffer(6)]],
    constant KernelParams& p [[buffer(7)]],
    constant BacktrackingParams& b [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup float4 high_scratch[256];
    threadgroup float4 low_scratch[256];
    if (leaf >= p.leaves) return;
    const uint tile_count = ObjectiveLeafTileCount(p);
    float3 high = float3(0.0f);
    float3 low = float3(0.0f);
    for (uint tile = tid; tile < tile_count; tile += 256) {
        const uint partial = 2 * (leaf * tile_count + tile);
        ObjectiveMergeExpansion(high, low, partials[partial].xyz,
                                 partials[partial + 1].xyz);
    }
    high_scratch[tid] = float4(high, 0.0f);
    low_scratch[tid] = float4(low, 0.0f);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) {
        const float3 statistics_high = high_scratch[0].xyz;
        const float3 statistics_low = low_scratch[0].xyz;
        const float3 statistics = statistics_high + statistics_low;
        output_weights[leaf] = statistics.z;
        float gradient_high = statistics_high.x;
        float gradient_low = statistics_low.x;
        float diagonal_high = p.leaf_method == 1
            ? statistics_high.z : statistics_high.y;
        float diagonal_low = p.leaf_method == 1
            ? statistics_low.z : statistics_low.y;
        if (b.normalize != 0) {
            const float2 gradient = BacktrackingDivideExpansion(
                gradient_high, gradient_low, p.total_weight);
            const float2 diagonal = BacktrackingDivideExpansion(
                diagonal_high, diagonal_low, p.total_weight);
            gradient_high = gradient.x;
            gradient_low = gradient.y;
            diagonal_high = diagonal.x;
            diagonal_low = diagonal.y;
        }
        // oracle_interface.h::AddRigdeRegulaizationIfNecessary is disabled by
        // default. Lambda still belongs to the diagonal when this flag is off.
        if (b.add_ridge != 0) {
            const float ridge = -p.l2 * current_values[leaf];
            ObjectiveAddExpansion(gradient_high, gradient_low, ridge);
            ObjectiveAddExpansion(gradient_high, gradient_low,
                fma(-p.l2, current_values[leaf], -ridge));
        }
        ObjectiveAddExpansion(diagonal_high, diagonal_low, p.l2);
        ObjectiveAddExpansion(gradient_high, gradient_low, gradient_noise[leaf].x);
        ObjectiveAddExpansion(gradient_high, gradient_low, gradient_noise[leaf].y);
        ObjectiveAddExpansion(diagonal_high, diagonal_low, diagonal_noise[leaf].x);
        ObjectiveAddExpansion(diagonal_high, diagonal_low, diagonal_noise[leaf].y);
        const float diagonal = diagonal_high + diagonal_low;
        // Combination may have negative YetiRank curvature. Its finite
        // nonpositive diagonal yields a zero direction below, as in CUDA.
        const bool grouped = p.objective == 12 || p.objective == 13 || p.objective == 19;
        if (!all(isfinite(statistics)) || (!grouped && statistics.y < 0.0f) || statistics.z < 0.0f
            || !isfinite(current_values[leaf]) || !isfinite(gradient_high)
            || !isfinite(gradient_low) || !isfinite(diagonal)) {
            directions[leaf] = ObjectiveInvalidValue();
            direction_dot[leaf] = float2(ObjectiveInvalidValue());
            return;
        }
        float direction = 0.0f;
        if (diagonal > 0.0f) {
            ObjectiveAddExpansion(diagonal_high, diagonal_low, 1e-20f);
            direction = BacktrackingDirection(gradient_high, gradient_low,
                                               diagonal_high, diagonal_low);
        }
        directions[leaf] = direction;
        float dot_high = gradient_high * direction;
        float dot_low = fma(gradient_high, direction, -dot_high);
        ObjectiveAddExpansion(dot_high, dot_low, gradient_low * direction);
        direction_dot[leaf] = float2(dot_high, dot_low);
    }
}
)METAL";
