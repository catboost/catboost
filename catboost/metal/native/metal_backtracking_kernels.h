#pragma once

// Append after CBMMetalObjectiveSource: this source reuses its compensated
// reductions and objective math. The runtime owns CUDA's scalar walker loop.
static const char* CBMMetalBacktrackingSource = R"METAL(

struct BacktrackingParams {
    float step;
    uint type;       // 1 AnyImprovement, 2 Armijo; acceptance is on the host.
    uint add_ridge;  // AddRidgeToTargetFunction; supported runtime default is 0.
    uint normalize;  // IsNormalize; supported runtime default is 0.
};

inline float2 BacktrackingDivideExpansion(float high, float low, float divisor) {
    const float quotient = high / divisor;
    const float remainder = fma(-quotient, divisor, high) + low;
    float quotient_high = quotient;
    float quotient_low = 0.0f;
    ObjectiveAddExpansion(quotient_high, quotient_low, remainder / divisor);
    return float2(quotient_high, quotient_low);
}

inline float BacktrackingDirection(float gradient_high, float gradient_low,
                                   float diagonal_high, float diagonal_low) {
    const float quotient = gradient_high / diagonal_high;
    const float remainder = fma(-quotient, diagonal_high, gradient_high)
        + gradient_low - quotient * diagonal_low;
    return quotient + remainder / diagonal_high;
}

// descent_helpers.cpp::UpdateMoveDirectionDiagonal computes a float direction
// from widened projected derivatives. Keep both float parts until division.
// Do not zero directions for sample-weight-empty leaves: CUDA regularizes
// the candidate point AFTER constructing its direction and Armijo dot product.
// Dispatch one group of exactly 256 threads per leaf.
// direction_dot contains one float2 (high,low) per leaf, summed by the host.
kernel void PrepareBacktrackingDirection(
    const device float4* partials [[buffer(0)]],
    const device float* current_values [[buffer(1)]],
    device float* directions [[buffer(2)]],
    device float* output_weights [[buffer(3)]],
    device float2* direction_dot [[buffer(4)]],
    constant KernelParams& p [[buffer(5)]],
    constant BacktrackingParams& b [[buffer(6)]],
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
        const float diagonal = diagonal_high + diagonal_low;
        const bool grouped = p.objective == 12 || p.objective == 13;
        if (!all(isfinite(statistics)) || (!grouped && statistics.y < 0.0f) || statistics.z < 0.0f
            || (grouped && p.leaf_method == 0 && statistics.z >= 1e-20f && diagonal <= 0.0f)
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

// CUDA MoveInOptimalDirection adds double(step)*float(direction) to the float
// current point and rounds once. Steps are exact powers of two; fma retains the
// corresponding single-rounding behavior. RegularizeImpl then zeros leaves
// whose ORIGINAL sample weight is below 1e-20 (not leaves with zero Hessian).
kernel void BuildBacktrackingCandidate(
    const device float* current_values [[buffer(0)]],
    const device float* directions [[buffer(1)]],
    const device float* output_weights [[buffer(2)]],
    device float* trial_values [[buffer(3)]],
    constant KernelParams& p [[buffer(4)]],
    constant BacktrackingParams& b [[buffer(5)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf < p.leaves) {
        trial_values[leaf] = output_weights[leaf] < 1e-20f ? 0.0f
            : fma(b.step, directions[leaf], current_values[leaf]);
    }
}

// These are the unnormalized CUDA target Scores. In particular RMSE's score
// has NO factor 1/2 even though its emitted gradient is target-prediction.
// Poisson can have a negative Score; it is never shifted to be nonnegative.
inline float BacktrackingRowScore(float target, float raw,
                                  constant KernelParams& p) {
    if (!isfinite(raw) || !isfinite(target)) return ObjectiveInvalidValue();
    if (p.objective == 0) {
        const float residual = target - raw;
        return residual * residual;
    }
    if (p.objective == 1 || p.objective == 2) {
        const float linear = raw >= 0.0f ? (1.0f - target) * raw : -target * raw;
        return linear + ObjectiveLogOnePlus(exp(-abs(raw)));
    }
    if (p.objective == 3) return exp(raw) - target * raw;
    const float residual = target - raw;
    if (p.objective == 4) {
        const float mismatch = abs(residual);
        return mismatch < p.objective_param ? 0.5f * mismatch * mismatch
            : p.objective_param * (mismatch - 0.5f * p.objective_param);
    }
    if (p.objective == 5) {
        const float multiplier = residual > 0.0f
            ? p.objective_param : 1.0f - p.objective_param;
        return multiplier == 0.0f ? 0.0f : (multiplier * residual) * residual;
    }
    if (p.objective >= 6) return AdditionalObjectiveValueDerivatives(target, raw, p.objective, p.objective_param).x;
    return ObjectiveInvalidValue();
}

// Return F = -sum(weight*Score) for ensemble_predictions + raw_point[leaf].
// Normalize data loss only if requested; ridge is -lambda/2*sum(point^2)
// AFTER normalization, precisely as the CUDA oracle does. The runtime uses
// flags=0 for the supported initial path. Each output is float2(high,low).
// Nonfinite trial objectives are intentionally preserved: the host rejects
// the trial and halves the step, without invalidating the last accepted point.
// Dispatch one-dimensional threadgroups, exactly 256 threads per group.
kernel void ReduceBacktrackingObjective(
    const device float* targets [[buffer(0)]],
    const device float* sample_weights [[buffer(1)]],
    const device float* ensemble_predictions [[buffer(2)]],
    const device uint* leaf_ids [[buffer(3)]],
    const device float* raw_point [[buffer(4)]],
    device float2* value_partials [[buffer(5)]],
    constant KernelParams& p [[buffer(6)]],
    constant BacktrackingParams& b [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 scratch[256];
    float high = 0.0f;
    float low = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float weight = sample_weights[row];
        if (weight == 0.0f) continue;
        const float raw = ensemble_predictions[row] + raw_point[leaf_ids[row]];
        const float score = BacktrackingRowScore(targets[row], raw, p);
        const float weighted_score = -weight * score;
        ObjectiveAddExpansion(high, low, weighted_score);
    }
    if (b.normalize != 0) {
        const float2 normalized = BacktrackingDivideExpansion(high, low, p.total_weight);
        high = normalized.x;
        low = normalized.y;
    }
    if (b.add_ridge != 0) {
        for (uint leaf = group * 256 + tid; leaf < p.leaves; leaf += groups * 256) {
            const float point = raw_point[leaf];
            ObjectiveAddExpansion(high, low, (-0.5f * p.l2 * point) * point);
        }
    }
    scratch[tid] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            high = scratch[tid].x;
            low = scratch[tid].y;
            ObjectiveAddExpansion(high, low, scratch[tid + offset].x);
            ObjectiveAddExpansion(high, low, scratch[tid + offset].y);
            scratch[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) value_partials[group] = scratch[0];
}
)METAL";
