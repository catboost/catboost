#pragma once

// CUDA pointwise_oracle.cpp returns full source dimensions (including the
// MultiClass anchor), with full Newton blocks except OneVsAll. Noise is added
// after lambda, independently to both triangles. LAPACK dposv('U') sees the
// row-major lower triangle; a positive factorization status leaves RHS intact.
// Float expansions retain callback doubles on Apple GPUs without float64.
static const char* CBMMetalVectorLangevinSource = R"METAL(
inline float2 VectorLangAdd(float2 a, float2 b) {
    float high = a.x, low = a.y;
    MulticlassAdd(high, low, b.x); MulticlassAdd(high, low, b.y);
    const float sum = high + low;
    return float2(sum, low - (sum - high));
}
inline float2 VectorLangMul(float2 a, float2 b) {
    const float high = a.x * b.x;
    return VectorLangAdd(float2(high, fma(a.x, b.x, -high)),
                        float2(a.x * b.y + a.y * b.x, a.y * b.y));
}
inline float2 VectorLangDiv(float2 a, float2 b) {
    const float high = a.x / b.x;
    const float2 residual = VectorLangAdd(a, -VectorLangMul(b, float2(high, 0)));
    return VectorLangAdd(float2(high, 0), float2((residual.x + residual.y) / b.x, 0));
}
inline float2 VectorLangSqrt(float2 a) {
    const float high = sqrt(a.x);
    const float2 residual = VectorLangAdd(a, -VectorLangMul(float2(high, 0), float2(high, 0)));
    return VectorLangAdd(float2(high, 0), float2((residual.x + residual.y) / (2 * high), 0));
}

kernel void VectorLangevinDirection(
    const device float* statistics [[buffer(0)]],
    const device float2* gradient_noise [[buffer(1)]],
    const device float2* hessian_noise [[buffer(2)]],
    device float2* workspace [[buffer(3)]],
    device float* directions [[buffer(4)]],
    device float4* dots [[buffer(5)]],
    device uint* status [[buffer(6)]],
    constant MulticlassParams& p [[buffer(7)]], uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const uint d = p.classes;
    const bool diagonal = p.leaf_method == 1 || p.objective == 1;
    const device float* stats = statistics + leaf * MulticlassStatsWidth(p);
    device float2* matrix = workspace + leaf * (d * d + 2 * d);
    device float2* gradient = matrix + d * d;
    device float2* solution = gradient + d;
    status[leaf] = 0;
    float2 sum = 0;
    for (uint i = 0; i < d; ++i) {
        const float2 clean = p.objective == 0 && i + 1 == d ? -sum : float2(stats[1 + i], 0);
        sum = VectorLangAdd(sum, clean);
        gradient[i] = VectorLangAdd(clean, gradient_noise[leaf * d + i]);
        solution[i] = gradient[i];
        if (!all(isfinite(gradient[i]))) status[leaf] = 1;
    }
    if (!isfinite(stats[0]) || stats[0] < 0) status[leaf] = 1;
    if (diagonal) {
        for (uint i = 0; i < d; ++i) {
            const float clean = p.leaf_method == 1 ? stats[0] : stats[1 + d + i];
            float2 h = VectorLangAdd(float2(clean, 0), float2(p.l2, 0));
            h = VectorLangAdd(h, hessian_noise[leaf * d + i]);
            if (!all(isfinite(h))) status[leaf] = 1;
            solution[i] = h.x + h.y > 0 ? VectorLangDiv(gradient[i], VectorLangAdd(h, float2(1e-20f, 0))) : float2(0);
        }
    } else {
        for (uint i = 0; i < d; ++i) for (uint j = 0; j <= i; ++j) {
            const float clean = p.objective == 0 ? stats[1 + d + i * (i + 1) / 2 + j]
                : (i == j ? stats[1 + d + i] : 0.0f);
            float2 h = VectorLangAdd(float2(clean, 0), float2(i == j ? p.l2 : 0, 0));
            h = VectorLangAdd(h, hessian_noise[leaf * d * d + i * d + j]);
            matrix[i * d + j] = h;
            if (!all(isfinite(h))) status[leaf] = 1;
        }
        bool positive = true;
        for (uint i = 0; i < d && positive; ++i) for (uint j = 0; j <= i; ++j) {
            float2 value = matrix[i * d + j];
            for (uint k = 0; k < j; ++k)
                value = VectorLangAdd(value, -VectorLangMul(matrix[i * d + k], matrix[j * d + k]));
            if (i == j) {
                if (!(value.x + value.y > 0)) { positive = false; break; }
                matrix[i * d + i] = VectorLangSqrt(value);
            } else matrix[i * d + j] = VectorLangDiv(value, matrix[j * d + j]);
        }
        // dposv does not call dpotrs after non-positive-definite dpotrf.
        if (positive) {
            for (uint i = 0; i < d; ++i) {
                float2 value = solution[i];
                for (uint j = 0; j < i; ++j) value = VectorLangAdd(value, -VectorLangMul(matrix[i * d + j], solution[j]));
                solution[i] = VectorLangDiv(value, matrix[i * d + i]);
            }
            for (uint reverse = 0; reverse < d; ++reverse) {
                const uint i = d - reverse - 1;
                float2 value = solution[i];
                for (uint j = i + 1; j < d; ++j) value = VectorLangAdd(value, -VectorLangMul(matrix[j * d + i], solution[j]));
                solution[i] = VectorLangDiv(value, matrix[i * d + i]);
            }
        }
    }
    int maximumExponent = -1024;
    for (uint i = 0; i < d; ++i) {
        const float direction = solution[i].x + solution[i].y;
        directions[leaf * d + i] = direction;
        if (!isfinite(direction)) status[leaf] = 1;
        if (gradient[i].x != 0 && direction != 0) {
            int left, right;
            frexp(gradient[i].x, left); frexp(direction, right);
            maximumExponent = max(maximumExponent, left + right);
        }
    }
    // CUDA evaluates Armijo dots in double. Preserve finite products whose
    // exponent exceeds float32 before the host restores their common scale.
    float2 dot = 0;
    if (maximumExponent == -1024) maximumExponent = 0;
    for (uint i = 0; i < d; ++i) {
        const float direction = directions[leaf * d + i];
        if (gradient[i].x == 0 || direction == 0) continue;
        int left, right;
        const float dm = frexp(direction, right);
        frexp(gradient[i].x, left);
        const float2 gm = float2(ldexp(gradient[i].x, -left), ldexp(gradient[i].y, -left));
        const float2 product = VectorLangMul(gm, float2(dm, 0));
        dot = VectorLangAdd(dot, float2(ldexp(product.x, left + right - maximumExponent),
                                       ldexp(product.y, left + right - maximumExponent)));
    }
    dots[leaf] = float4(dot, float(maximumExponent), 0);
    if (!all(isfinite(dot))) status[leaf] = 1;
}

kernel void VectorLangevinCandidate(
    const device float* current [[buffer(0)]], const device float* direction [[buffer(1)]],
    const device float* statistics [[buffer(2)]], device float* candidate [[buffer(3)]],
    constant MulticlassParams& p [[buffer(4)]], constant MulticlassBacktrackingParams& b [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.leaves * p.classes) return;
    const uint leaf = index / p.classes;
    const float2 point = VectorLangAdd(float2(current[index], 0),
                                     VectorLangMul(float2(b.step, 0), float2(direction[index], 0)));
    candidate[index] = statistics[leaf * MulticlassStatsWidth(p)] < p.min_leaf_weight ? 0 : point.x + point.y;
}

kernel void VectorLangevinGauge(
    const device float* source [[buffer(0)]], device float* cursor_values [[buffer(1)]],
    constant MulticlassParams& p [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    const uint d = MulticlassDimension(p);
    if (index >= p.leaves * d) return;
    const uint leaf = index / d, dim = index % d;
    cursor_values[index] = source[leaf * p.classes + dim]
        - (p.objective == 0 ? source[leaf * p.classes + p.classes - 1] : 0.0f);
}
)METAL";
