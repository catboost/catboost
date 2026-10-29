#pragma once

// Given-edge PairLogit pointwise target, from cuda/targets/kernel/pair_logit.cu.
// PairLogitPairwise additionally needs a leaf Laplacian and its separate
// regularization/solver; a diagonal projection does not implement that loss.
static const char* CBMMetalPairwiseSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct PairwiseParams {
    uint rows, pairs, objective, apply_leaf_values;
    uint leaves, reserved0, reserved1, reserved2;
};

inline float PairwiseLogOnePlus(float value) {
    // MSL has no log1p. Correct the rounded addition for value > -1,
    // retaining tiny signed changes when 1+value rounds to exactly one.
    const float rounded = 1.0f + value;
    return rounded == 1.0f ? value : log(rounded) * (value / (rounded - 1.0f));
}

inline void PairwiseAccumulate4(thread float4& high, thread float4& low, float4 value) {
    const float4 sum = high + value;
    const float4 virtual_value = sum - high;
    const float4 error = (high - (sum - virtual_value)) + (value - virtual_value) + low;
    high = sum + error;
    low = error - (high - sum);
}

kernel void PreparePairwisePoint(const device float* cursor [[buffer(0)]],
                                 const device float* raw_leaf_values [[buffer(1)]],
                                 const device uint* leaf_ids [[buffer(2)]],
                                 device float* point [[buffer(3)]],
                                 constant PairwiseParams& p [[buffer(4)]],
                                 uint row [[thread_position_in_grid]]) {
    if (row < p.rows) point[row] = cursor[row]
        + (p.apply_leaf_values ? raw_leaf_values[leaf_ids[row]] : 0.0f);
}

// Supplied weights are used literally, with no group/object multiplication.
// Generated pairs may already contain a group-derived weight; do not reweight.
// Outputs: (winner ascent derivative, edge curvature, positive loss, weight).
kernel void PairLogitEdgeDerivatives(const device float* point [[buffer(0)]],
                                     const device uint* winners [[buffer(1)]],
                                     const device uint* losers [[buffer(2)]],
                                     const device float* pair_weights [[buffer(3)]],
                                     device float4* edge_values [[buffer(4)]],
                                     constant PairwiseParams& p [[buffer(5)]],
                                     uint edge [[thread_position_in_grid]]) {
    if (edge >= p.pairs) return;
    const float weight = pair_weights[edge];
    if (weight == 0.0f) { edge_values[edge] = float4(0.0f); return; }
    const float difference = point[winners[edge]] - point[losers[edge]];
    const float exponential = exp(-abs(difference));
    // Stable sigmoid preserves CUDA's float32 saturation of p and 1-p.
    const float probability = clamp(difference >= 0.0f
        ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential), 1e-40f, 1.0f);
    const float gradient = weight * (1.0f - probability);
    const float curvature = weight * probability * (1.0f - probability);
    // Log-domain softplus avoids overflow and diff-log(1+exp(diff))
    // cancellation for confident winners in the original CUDA expression.
    const float loss = weight * (max(-difference, 0.0f) + PairwiseLogOnePlus(exponential));
    edge_values[edge] = float4(gradient, curvature, loss, weight);
}

// Deterministic incidence CSR: each edge appears twice, sign +1 at winner
// and -1 at loser, in stable input-edge order. No float atomics are needed.
// Dispatch exactly rows full256-thread groups. The same positive curvature
// goes to both endpoints (CUDA's POINTWISE diagonal approximation).
kernel void ReducePairwiseRows(const device uint* row_offsets [[buffer(0)]],
                               const device uint* incident_edges [[buffer(1)]],
                               const device int* incident_signs [[buffer(2)]],
                               const device float4* edge_values [[buffer(3)]],
                               device float* gradients [[buffer(4)]],
                               device float* curvature [[buffer(5)]],
                               device float* incident_weights [[buffer(6)]],
                               constant PairwiseParams& p [[buffer(7)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint row [[threadgroup_position_in_grid]]) {
    if (row >= p.rows) return;
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint index = row_offsets[row] + tid; index < row_offsets[row + 1]; index += 256) {
        const float4 edge = edge_values[incident_edges[index]];
        PairwiseAccumulate4(high, low, float4(float(incident_signs[index]) * edge.x, edge.y, edge.w, 0.0f));
    }
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = high_parts[tid]; low = low_parts[tid];
            PairwiseAccumulate4(high, low, high_parts[tid + stride]);
            PairwiseAccumulate4(high, low, low_parts[tid + stride]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float4 total = high_parts[0] + low_parts[0];
        gradients[row] = total.x;
        curvature[row] = total.y;
        incident_weights[row] = total.z;
    }
}

// Outputs partial mean positive loss and mean edge weight. Host sums in
// double, then multiplies by pairs for the unnormalized leaf oracle. The
// metric ratio is sum(loss)/sum(edge weights), not sum incident row weights.
kernel void ReducePairwiseObjective(const device float4* edge_values [[buffer(0)]],
                                    device float2* partials [[buffer(1)]],
                                    constant PairwiseParams& p [[buffer(2)]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 scratch[256];
    float2 sum = float2(0.0f), correction = float2(0.0f);
    for (uint edge = group * 256 + tid; edge < p.pairs; edge += groups * 256) {
        const float4 entry = edge_values[edge];
        const float2 value = entry.zw / float(p.pairs) - correction;
        const float2 next = sum + value;
        correction = (next - sum) - value;
        sum = next;
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) scratch[tid] += scratch[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = scratch[0];
}

inline float PairwiseExpm1(float value) {
    // The loss-difference path only calls this for |value| <= 0.5. A Taylor
    // polynomial avoids losing the update when exp(value) rounds to one.
    return value * (1.0f + value * (0.5f + value * (1.0f / 6.0f + value *
        (1.0f / 24.0f + value * (1.0f / 120.0f + value * (1.0f / 720.0f +
        value * (1.0f / 5040.0f + value * (1.0f / 40320.0f + value *
        (1.0f / 362880.0f + value / 3628800.0f)))))))));
}

// Return an exact float expansion of a-b, provided the difference is finite.
inline float2 PairwiseDifference(float a, float b) {
    const float high = a - b;
    const float virtual_b = a - high;
    return float2(high, (a - (high + virtual_b)) + (virtual_b - b));
}

// Backtracking must resolve improvements much smaller than the absolute
// objective's float32 ULP. For a small margin change h, use the identity
// softplus(-d)-softplus(-(d+h)) = -log1p(sigmoid(-d)*expm1(-h)), then reduce
// those signed differences with an expansion. The point additions still use
// float32, exactly as the derivative path and the eventual cursor update do.
kernel void ReducePairwiseObjectiveDifference(const device float* cursor [[buffer(0)]],
    const device float* current_leaves [[buffer(1)]], const device float* trial_leaves [[buffer(2)]],
    const device uint* leaf_ids [[buffer(3)]], const device uint* winners [[buffer(4)]],
    const device uint* losers [[buffer(5)]], const device float* weights [[buffer(6)]],
    device float2* partials [[buffer(7)]], constant PairwiseParams& p [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]], uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint edge = group * 256 + tid; edge < p.pairs; edge += groups * 256) {
        if (weights[edge] == 0.0f) continue;
        const uint winner = winners[edge], loser = losers[edge];
        const float current_winner = cursor[winner] + current_leaves[leaf_ids[winner]];
        const float current_loser = cursor[loser] + current_leaves[leaf_ids[loser]];
        const float trial_winner = cursor[winner] + trial_leaves[leaf_ids[winner]];
        const float trial_loser = cursor[loser] + trial_leaves[leaf_ids[loser]];
        if (!all(isfinite(float4(current_winner, current_loser, trial_winner, trial_loser)))) {
            // A nonfinite trial is rejected without making accepted-state
            // validation sticky; subsequent halved trials can still succeed.
            PairwiseAccumulate4(high, low, float4(as_type<float>(0x7fc00000u), 0.0f, 0.0f, 0.0f));
            continue;
        }
        const float2 before = PairwiseDifference(current_winner, current_loser);
        const float2 after = PairwiseDifference(trial_winner, trial_loser);
        if (!isfinite(before.x) || !isfinite(after.x)) {
            // Finite endpoints can overflow their margin subtraction. A
            // positive saturated margin has zero representable softplus;
            // negative overflow is a nonfinite loss and rejects the trial.
            const float improvement = (isinf(before.x) && before.x < 0.0f)
                    || (isinf(after.x) && after.x < 0.0f)
                ? as_type<float>(0x7fc00000u)
                : (isinf(before.x) ? -(max(-after.x, 0.0f) + PairwiseLogOnePlus(exp(-abs(after.x))))
                    : max(-before.x, 0.0f) + PairwiseLogOnePlus(exp(-abs(before.x))));
            PairwiseAccumulate4(high, low, float4(weights[edge] * improvement / float(p.pairs), 0.0f, 0.0f, 0.0f));
            continue;
        }
        const float2 delta_high = PairwiseDifference(after.x, before.x);
        const float delta = delta_high.x + ((after.y - before.y) + delta_high.y);
        float improvement;
        if (abs(delta) <= 0.5f) {
            const float exponential = exp(-abs(before.x));
            float probability = before.x >= 0.0f
                ? exponential / (1.0f + exponential) : 1.0f / (1.0f + exponential);
            probability -= before.y * probability * (1.0f - probability);
            improvement = -PairwiseLogOnePlus(probability * PairwiseExpm1(-delta));
        } else {
            // Separate the linear terms so two large negative margins do
            // not discard a much smaller (but still > 0.5) improvement.
            const float linear = before.x < 0.0f
                ? (after.x < 0.0f ? delta : -before.x - before.y)
                : (after.x < 0.0f ? after.x + after.y : 0.0f);
            improvement = linear + (PairwiseLogOnePlus(exp(-abs(before.x)))
                - PairwiseLogOnePlus(exp(-abs(after.x))));
        }
        PairwiseAccumulate4(high, low, float4(weights[edge] * improvement / float(p.pairs), 0.0f, 0.0f, 0.0f));
    }
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = high_parts[tid]; low = low_parts[tid];
            PairwiseAccumulate4(high, low, high_parts[tid + stride]);
            PairwiseAccumulate4(high, low, low_parts[tid + stride]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = float2(high_parts[0].x, low_parts[0].x);
}

// The persistent runtime validates arbitrary trial points before indexing or
// propagating nonfinite values. Status stays sticky until its host owner has
// observed the completed command and explicitly cleared it.
kernel void PrepareValidatedPairwisePoint(const device float* cursor [[buffer(0)]],
    const device float* raw_leaf_values [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    device float* point [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant PairwiseParams& p [[buffer(5)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint leaf = p.apply_leaf_values ? leaf_ids[row] : 0;
    if (p.apply_leaf_values && leaf >= p.leaves) {
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
        point[row] = 0.0f;
        return;
    }
    const float value = cursor[row] + (p.apply_leaf_values ? raw_leaf_values[leaf] : 0.0f);
    if (!p.reserved1 && !isfinite(value)) atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
    point[row] = value;
}

kernel void ValidatePairwiseStatistics(const device float4* edges [[buffer(0)]],
    const device float* gradients [[buffer(1)]], const device float* hessian [[buffer(2)]],
    const device float* incident_weights [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant PairwiseParams& p [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    if (index < p.pairs) {
        const float4 edge = edges[index];
        if (!p.reserved1 && (!all(isfinite(edge)) || any(edge < 0.0f)))
            atomic_fetch_or_explicit(status, 2u, memory_order_relaxed);
    }
    if (!p.reserved1 && index < p.rows && (!isfinite(gradients[index]) || !isfinite(hessian[index])
            || !isfinite(incident_weights[index]) || hessian[index] < 0.0f || incident_weights[index] < 0.0f))
        atomic_fetch_or_explicit(status, 4u, memory_order_relaxed);
}

// NeedZeroAverage in CUDA's doc-parallel leaf estimator uses every actual
// leaf, including empty leaves, before tree shrinkage. Divide before summing
// to keep a mean of finite same-sign values finite near FLT_MAX.
kernel void ReducePairwiseLeafMean(const device float* leaves [[buffer(0)]],
    device float2* partials [[buffer(1)]], device atomic_uint* status [[buffer(2)]],
    constant PairwiseParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint leaf = group * 256 + tid; leaf < p.leaves; leaf += groups * 256) {
        const float value = leaves[leaf];
        if (!isfinite(value)) atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
        PairwiseAccumulate4(high, low, float4(value / float(p.leaves), 0.0f, 0.0f, 0.0f));
    }
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = high_parts[tid]; low = low_parts[tid];
            PairwiseAccumulate4(high, low, high_parts[tid + stride]);
            PairwiseAccumulate4(high, low, low_parts[tid + stride]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = float2(high_parts[0].x, low_parts[0].x);
}

kernel void FinalizePairwiseLeafMean(const device float2* partials [[buffer(0)]],
    device float* mean [[buffer(1)]], constant PairwiseParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint index = tid; index < p.reserved0; index += 256) {
        PairwiseAccumulate4(high, low, float4(partials[index].x, 0.0f, 0.0f, 0.0f));
        PairwiseAccumulate4(high, low, float4(partials[index].y, 0.0f, 0.0f, 0.0f));
    }
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = high_parts[tid]; low = low_parts[tid];
            PairwiseAccumulate4(high, low, high_parts[tid + stride]);
            PairwiseAccumulate4(high, low, low_parts[tid + stride]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) mean[0] = high_parts[0].x + low_parts[0].x;
}

kernel void CenterPairwiseLeafValues(device float* leaves [[buffer(0)]],
    const device float* mean [[buffer(1)]], device atomic_uint* status [[buffer(2)]],
    constant PairwiseParams& p [[buffer(3)]], uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const float value = leaves[leaf] - mean[0];
    if (!isfinite(value)) atomic_fetch_or_explicit(status, 8u, memory_order_relaxed);
    leaves[leaf] = value;
}
)METAL";
