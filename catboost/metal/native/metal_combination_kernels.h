#pragma once

// Append after the scalar objective and query kernels. Each component keeps
// its own units: neither derivatives nor Hessians are normalized by its metric
// denominator. CUDA sums query components first and scalar components second.
static const char* CBMMetalCombinationSource = R"METAL(
struct CombinationParams {
    uint rows, objective, apply_shift, leaves;
    float coefficient, param, border;
    uint loss_groups;
};

// A line-search trial may overflow even though the accepted cursor is finite.
// Preserve a rejection marker while supplying a finite throwaway point to the
// strict Yeti target. Its seed is still consumed, as in CUDA's full oracle call.
kernel void PrepareCombinationYetiPoint(const device float* point [[buffer(0)]],
    device float* safe_point [[buffer(1)]], device atomic_uint* trial_status [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant CombinationParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const bool finite = isfinite(point[row]);
    safe_point[row] = finite ? point[row] : 0.0f;
    if (!finite) {
        if (p.apply_shift) atomic_fetch_or_explicit(trial_status, 1u, memory_order_relaxed);
        else atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
}

inline float CombinationRowLoss(float target, float raw, uint objective, float param) {
    if (!isfinite(raw) || !isfinite(target)) return ObjectiveInvalidValue();
    const float residual = target - raw;
    if (objective == 0) return residual * residual;
    if (objective == 1 || objective == 2) {
        const float linear = raw >= 0.0f ? (1.0f - target) * raw : -target * raw;
        return linear + ObjectiveLogOnePlus(exp(-abs(raw)));
    }
    if (objective == 3) return exp(raw) - target * raw;
    if (objective == 4) return abs(residual) < param ? .5f * residual * residual
        : param * (abs(residual) - .5f * param);
    if (objective == 5) return (residual > 0.0f ? param : 1.0f - param) * residual * residual;
    return AdditionalObjectiveValueDerivatives(target, raw, objective, param).x;
}

kernel void CombinationPointwise(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* point [[buffer(2)]],
    device float* gradient [[buffer(3)]], device float* hessian [[buffer(4)]],
    device float2* row_stats [[buffer(5)]], constant CombinationParams& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    float target = targets[row];
    if (p.objective == 1) target = float(target > p.border);
    const bool valid = isfinite(target) && (p.objective != 2 || (target >= 0 && target <= 1))
        && ((p.objective != 3 && p.objective != 7) || target >= 0);
    const float weight = weights[row];
    float2 value = valid ? ObjectiveGradientAndHessian(target, weight, point[row], p.objective, p.param)
        : float2(ObjectiveInvalidValue());
    gradient[row] = value.x; hessian[row] = value.y;
    row_stats[row] = float2(!valid ? ObjectiveInvalidValue()
        : weight == 0 ? 0 : weight * CombinationRowLoss(target, point[row], p.objective, p.param), weight);
}

kernel void AccumulateCombination(const device float* gradient [[buffer(0)]],
    const device float* hessian [[buffer(1)]], const device float* gradient_weights [[buffer(2)]],
    device float* total_gradient [[buffer(3)]], device float* total_hessian [[buffer(4)]],
    device float* total_weights [[buffer(5)]], constant CombinationParams& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    // Spell out multiply then add, matching MultiplyAddVector's CUDA float
    // arithmetic (the library is compiled with fast math disabled).
    total_gradient[row] += p.coefficient * gradient[row];
    total_hessian[row] += p.coefficient * hessian[row];
    total_weights[row] += p.coefficient * gradient_weights[row];
}

kernel void ReduceCombinationStats(const device float2* stats [[buffer(0)]],
    device float4* partials [[buffer(1)]], constant CombinationParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    threadgroup float4 high_scratch[256], low_scratch[256];
    float4 high = 0, low = 0;
    for (uint row = group * 256 + tid; row < p.rows; row += p.loss_groups * 256)
        QuerywiseAccumulate(float4(stats[row], 0, 0), high, low);
    high_scratch[tid] = high; low_scratch[tid] = low;
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (!tid) partials[group] = float4(high_scratch[0].xy, low_scratch[0].xy);
}

kernel void ValidateCombinationStatistics(const device float* gradient [[buffer(0)]],
    const device float* hessian [[buffer(1)]], const device float* gradient_weights [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant CombinationParams& p [[buffer(4)]],
    uint row [[thread_position_in_grid]]) {
    if (row < p.rows && (!isfinite(gradient[row]) || !isfinite(hessian[row]) || !isfinite(gradient_weights[row])))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}
)METAL";
