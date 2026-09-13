#pragma once

// Append after metal_stdlib / using namespace metal, and before the objective
// dispatch helpers. This source is independent of KernelParams and weighting.
static const char* CBMMetalAdditionalObjectiveSource = R"METAL(

// Every supported exponent is nonnegative. Spell out 0^0 because CUDA's Lq
// q=1 gradient and q=2 Hessian require it to be exactly one at residual zero.
inline float AdditionalObjectivePower(float magnitude, float exponent) {
    if (exponent == 0.0f) return 1.0f;
    if (magnitude == 0.0f) return 0.0f;
    return pow(magnitude, exponent);
}

// Unweighted (positive loss, negative first derivative, positive curvature),
// matching targets/kernel/pointwise_targets.cu::{Score,Der,Der2}. The caller
// must skip zero-weight observations BEFORE evaluating this helper, then apply
// the sample weight. Invalid inputs and genuine arithmetic overflow remain
// nonfinite so the runtime can report them; no exponential clipping is used.
inline float3 AdditionalObjectiveValueDerivatives(float target, float raw,
                                                  uint objective, float param) {
    const float invalid = as_type<float>(0x7fc00000u);
    if (!isfinite(target) || !isfinite(raw)) return float3(invalid);

    if (objective == 6) {
        // TLqTarget, lines 197-217. CUDA sign(0) is -1, so q=1 at an
        // exact match has gradient -1. For 1<=q<2 CUDA returns curvature 1;
        // the host must reject Newton estimation for that parameter range.
        const float q = param;
        if (!isfinite(q) || q < 1.0f) return float3(invalid);
        const float residual = target - raw;
        const float magnitude = abs(residual);
        const float sign = residual > 0.0f ? 1.0f : -1.0f;
        const float loss = AdditionalObjectivePower(magnitude, q);
        const float gradient = magnitude == 0.0f && q > 1.0f ? 0.0f
            : (q * sign) * AdditionalObjectivePower(magnitude, q - 1.0f);
        const float hessian = q < 2.0f ? 1.0f
            : (magnitude == 0.0f && q > 2.0f ? 0.0f
               : (q * (q - 1.0f)) * AdditionalObjectivePower(magnitude, q - 2.0f));
        return float3(loss, gradient, hessian);
    }

    if (objective == 7) {
        // TTweedieTarget, lines 34-57. Consume the actual variance_power;
        // do not reproduce CUDA host forwarding of the unrelated alpha.
        const float variance_power = param;
        if (!isfinite(variance_power) || variance_power <= 1.0f
            || variance_power >= 2.0f || target < 0.0f) return float3(invalid);
        const float first_power = 1.0f - variance_power;
        const float second_power = 2.0f - variance_power;
        // A zero target contributes no first term even when its unused
        // exponential would overflow (for a large negative raw prediction).
        const float first = target == 0.0f ? 0.0f : target * exp(first_power * raw);
        const float second = exp(second_power * raw);
        const float loss = -first / first_power + second / second_power;
        const float gradient = first - second;
        const float hessian = -first * first_power + second * second_power;
        return float3(loss, gradient, hessian);
    }

    if (objective == 8 || objective == 9 || objective == 10) {
        // TLogLinQuantileTarget, lines 123-143; TQuantileTarget, lines 11-31.
        // MAE training uses Quantile(alpha=.5), hence HALF absolute loss and
        // +/-0.5 gradients. Full MAE is a reporting conversion only; retain
        // this scale for leaf estimation, regularization and backtracking.
        const float alpha = objective == 10 ? 0.5f : param;
        if (!isfinite(alpha) || alpha < 0.0f || alpha > 1.0f) return float3(invalid);
        const float prediction = objective == 8 ? exp(raw) : raw;
        const float residual = target - prediction;
        // Equality takes CUDA's negative branch, including exact median ties.
        const float multiplier = residual > 0.0f ? alpha : -(1.0f - alpha);
        if (multiplier == 0.0f) return float3(0.0f);
        return float3(multiplier * residual,
                      objective == 8 ? multiplier * prediction : multiplier, 0.0f);
    }

    if (objective == 11) {
        // TMAPETarget, lines 146-158. Negative and zero finite targets are
        // supported; the denominator is max(1,abs(target)), never abs(target)
        // alone. Equality uses -1/denominator and the Hessian is zero.
        const float residual = target - raw;
        const float denominator = max(1.0f, abs(target));
        const float gradient = residual > 0.0f ? 1.0f / denominator : -1.0f / denominator;
        return float3(abs(residual) / denominator, gradient, 0.0f);
    }
    return float3(invalid);
}
)METAL";
