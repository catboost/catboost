#pragma once

// CUDA provenance and runtime integration contract: ../QUERYWISE_PORT.md.
// This source is independent of the scalar KernelParams ABI.
static const char* CBMMetalQuerywiseSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct QuerywiseParams {
    uint rows;
    uint groups;
    uint objective;          // 12: QueryRMSE, 13: QuerySoftMax
    uint apply_leaf_values;
    float beta;
    float lambda;
    uint leaves;
    uint reserved;           // bit 0: mark invalid Newton structure curvature
};

struct QuerywiseProjectionParams {
    uint rows, leaves, tiles, leaf_method; // leaf_method 1 ignores curvature
};

// Keep both components through every lane and tile merge. Per-lane Kahan
// followed by an ordinary float tree loses the 1 in {2^24, 1, -2^24}.
inline void QuerywiseAccumulate(float value, thread float& high, thread float& low) {
    const float sum = high + value;
    const float value_part = sum - high;
    const float error = (high - (sum - value_part)) + (value - value_part);
    const float tail = low + error;
    const float next = sum + tail;
    const float tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

inline void QuerywiseAccumulate(float4 value, thread float4& high, thread float4& low) {
    const float4 sum = high + value;
    const float4 value_part = sum - high;
    const float4 error = (high - (sum - value_part)) + (value - value_part);
    const float4 tail = low + error;
    const float4 next = sum + tail;
    const float4 tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

inline void QuerywiseReduceExpansions(threadgroup float4* high_scratch,
                                      threadgroup float4* low_scratch, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            float4 high = high_scratch[tid], low = low_scratch[tid];
            QuerywiseAccumulate(high_scratch[tid + stride], high, low);
            QuerywiseAccumulate(low_scratch[tid + stride], high, low);
            high_scratch[tid] = high;
            low_scratch[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline float2 QuerywiseDivideExpansion(float high, float low,
                                       float denominator_high, float denominator_low) {
    const float quotient = high / denominator_high;
    const float remainder = fma(-quotient, denominator_high, high)
        + low - quotient * denominator_low;
    float result_high = quotient, result_low = 0.0f;
    QuerywiseAccumulate(remainder / denominator_high, result_high, result_low);
    return float2(result_high, result_low);
}

kernel void ResetQuerywiseLeafIds(device uint* leaf_ids [[buffer(0)]],
                                  constant QuerywiseParams& p [[buffer(1)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row < p.rows) leaf_ids[row] = 0;
}

// Validate row Hessians before using them as Newton histogram weights. A NaN
// gradient marker alone is insufficient: L2 scoring skips negative-weight
// children before inspecting their gradient. Host clears this 4-byte flag.
kernel void ValidateQuerywiseStructureCurvature(
    const device float* curvature [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant QuerywiseParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row < p.rows && (!isfinite(curvature[row]) || curvature[row] < 0.0f))
        atomic_fetch_or_explicit(status, 1u, memory_order_relaxed);
}

// Preserve ORIGINAL query row order even when the leaf partition interleaves
// a query. Raw leaf values are the unshrunk trial point, not a tree to append.
kernel void PrepareQuerywisePoint(const device float* cursor [[buffer(0)]],
                                  const device float* raw_leaf_values [[buffer(1)]],
                                  const device uint* leaf_ids [[buffer(2)]],
                                  device float* point [[buffer(3)]],
                                  constant QuerywiseParams& p [[buffer(4)]],
                                  uint row [[thread_position_in_grid]]) {
    if (row < p.rows) {
        point[row] = cursor[row] + (p.apply_leaf_values ? raw_leaf_values[leaf_ids[row]] : 0.0f);
    }
}

// Dispatch one complete 256-thread threadgroup per query. Offsets have
// groups+1 entries, start at zero, end at rows, and increase strictly.
// All outputs are in original row order. Stats are positive loss and metric
// denominator, without leaf regularization or a final square root.
kernel void QueryRmseDerivatives(const device float* targets [[buffer(0)]],
                                 const device float* weights [[buffer(1)]],
                                 const device float* point [[buffer(2)]],
                                 const device uint* offsets [[buffer(3)]],
                                 device float* gradients [[buffer(4)]],
                                 device float* curvature [[buffer(5)]],
                                 device float2* query_stats [[buffer(6)]],
                                 constant QuerywiseParams& p [[buffer(7)]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint query [[threadgroup_position_in_grid]]) {
    if (query >= p.groups) return;
    threadgroup float4 high_scratch[256], low_scratch[256];
    const uint begin = offsets[query], end = offsets[query + 1];
    float sum = 0.0f, correction = 0.0f;
    float mass = 0.0f, mass_correction = 0.0f;
    for (uint row = begin + tid; row < end; row += 256) {
        const float weight = weights[row];
        if (weight > 0.0f) {
            QuerywiseAccumulate(weight * (targets[row] - point[row]), sum, correction);
            QuerywiseAccumulate(weight, mass, mass_correction);
        }
    }
    high_scratch[tid] = float4(sum, mass, 0, 0);
    low_scratch[tid] = float4(correction, mass_correction, 0, 0);
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    const float total_mass = high_scratch[0].y + low_scratch[0].y;
    const float2 mean = total_mass != 0.0f ? QuerywiseDivideExpansion(
        high_scratch[0].x, low_scratch[0].x, high_scratch[0].y, low_scratch[0].y) : float2(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float loss = 0.0f, loss_correction = 0.0f;
    for (uint row = begin + tid; row < end; row += 256) {
        const float weight = weights[row];
        // A zero-weight extreme row can have an overflowing y-a. It must
        // contribute exactly zero without evaluating 0*infinity.
        if (weight == 0.0f) { gradients[row] = 0.0f; curvature[row] = 0.0f; continue; }
        float residual_high = targets[row] - point[row], residual_low = 0.0f;
        QuerywiseAccumulate(-mean.x, residual_high, residual_low);
        QuerywiseAccumulate(-mean.y, residual_high, residual_low);
        const float residual = residual_high + residual_low;
        gradients[row] = weight * residual;
        // Intentionally CUDA's diagonal approximation, not w*(1-w/sum(w)).
        curvature[row] = weight;
        QuerywiseAccumulate(weight * residual * residual, loss, loss_correction);
    }
    high_scratch[tid] = float4(loss, 0, 0, 0);
    low_scratch[tid] = float4(loss_correction, 0, 0, 0);
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) query_stats[query] = float2(high_scratch[0].x + low_scratch[0].x, total_mass);
}

// CUDA query_softmax.cu: p_i = w_i*exp(beta*a_i)/sum_j(...),
// A=sum(w_i*t_i), gradient=beta*(w_i*t_i-A*p_i), and
// curvature=beta*A*(beta*p_i*(1-p_i)+lambda) on positive-weight rows.
// Lambda is a curvature regularizer ONLY, not an objective penalty.
kernel void QuerySoftMaxDerivatives(const device float* targets [[buffer(0)]],
                                    const device float* weights [[buffer(1)]],
                                    const device float* point [[buffer(2)]],
                                    const device uint* offsets [[buffer(3)]],
                                    device float* gradients [[buffer(4)]],
                                    device float* curvature [[buffer(5)]],
                                    device float2* query_stats [[buffer(6)]],
                                    constant QuerywiseParams& p [[buffer(7)]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint query [[threadgroup_position_in_grid]]) {
    if (query >= p.groups) return;
    threadgroup float4 high_scratch[256], low_scratch[256];
    threadgroup float max_scratch[256];
    const uint begin = offsets[query], end = offsets[query + 1];
    float maximum = -INFINITY;
    float target_mass = 0.0f, target_correction = 0.0f;
    for (uint row = begin + tid; row < end; row += 256) {
        const float weight = weights[row];
        if (weight > 0.0f) {
            maximum = max(maximum, p.beta < 0.0f ? -point[row] : point[row]);
            QuerywiseAccumulate(weight * targets[row], target_mass, target_correction);
        }
    }
    max_scratch[tid] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            max_scratch[tid] = max(max_scratch[tid], max_scratch[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum = p.beta < 0.0f ? -max_scratch[0] : max_scratch[0];
    high_scratch[tid] = float4(target_mass, 0, 0, 0);
    low_scratch[tid] = float4(target_correction, 0, 0, 0);
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    const float target_high = high_scratch[0].x, target_low = low_scratch[0].x;
    target_mass = target_high + target_low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float exponent_sum = 0.0f, exponent_correction = 0.0f;
    for (uint row = begin + tid; row < end; row += 256) {
        const float weight = weights[row];
        if (weight > 0.0f) {
            QuerywiseAccumulate(weight * exp(p.beta * (point[row] - maximum)),
                exponent_sum, exponent_correction);
        }
    }
    high_scratch[tid] = float4(exponent_sum, 0, 0, 0);
    low_scratch[tid] = float4(exponent_correction, 0, 0, 0);
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    exponent_sum = high_scratch[0].x + low_scratch[0].x;
    const float log_sum = exponent_sum > 0.0f ? log(exponent_sum) : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float loss = 0.0f, loss_correction = 0.0f;
    for (uint row = begin + tid; row < end; row += 256) {
        const float weight = weights[row], target = targets[row];
        if (weight == 0.0f) { gradients[row] = 0.0f; curvature[row] = 0.0f; continue; }
        const bool active = weight > 0.0f && target_mass > 0.0f;
        const float probability = active
            ? weight * exp(p.beta * (point[row] - maximum)) / exponent_sum : 0.0f;
        float gradient_high = weight * target, gradient_low = 0.0f;
        if (active) {
            const float contribution = -target_high * probability;
            QuerywiseAccumulate(contribution, gradient_high, gradient_low);
            QuerywiseAccumulate(fma(-target_high, probability, -contribution), gradient_high, gradient_low);
            QuerywiseAccumulate(-target_low * probability, gradient_high, gradient_low);
        }
        gradients[row] = p.beta * (gradient_high + gradient_low);
        const float hessian = active
            ? p.beta * target_mass * (p.beta * probability * (1.0f - probability) + p.lambda) : 0.0f;
        curvature[row] = hessian;
        if ((p.reserved & 1u) && (!isfinite(hessian) || hessian < 0.0f))
            gradients[row] = as_type<float>(0x7fc00000u);
        if (weight > 0.0f && target > 0.0f) {
            // Log-domain evaluation preserves a finite loss when float32 p_i
            // underflows. CUDA uses log(p_i), which can become -infinity.
            const float negative_log_probability = log_sum - log(weight) - (p.beta * (point[row] - maximum));
            QuerywiseAccumulate(weight * target * negative_log_probability, loss, loss_correction);
        }
    }
    high_scratch[tid] = float4(loss, 0, 0, 0);
    low_scratch[tid] = float4(loss_correction, 0, 0, 0);
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) query_stats[query] = float2(high_scratch[0].x + low_scratch[0].x, target_mass);
}

// Project PRECOMPUTED original-row grouped derivatives through a leaf
// partition. No query is renormalized here and no sampling weight is applied.
// Dispatch (tiles,leaves) full 256-thread groups. Compatible with
// ReduceLeafObjectivePartials / EstimateNewtonLeafValues: consecutive float4
// high/low pairs hold (gradient,curvature,original_weight,0).
kernel void ReduceQuerywiseLeafPartials(
    const device float* gradients [[buffer(0)]],
    const device float* curvature [[buffer(1)]],
    const device float* original_weights [[buffer(2)]],
    const device uint* row_indices [[buffer(3)]],
    const device uint* partition_offsets [[buffer(4)]],
    device float4* partials [[buffer(5)]],
    constant QuerywiseProjectionParams& p [[buffer(6)]],
    uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local.x, leaf = group.y, tile = group.x;
    if (leaf >= p.leaves || tile >= p.tiles) return;
    threadgroup float4 high_scratch[256], low_scratch[256];
    float4 high = float4(0), low = float4(0);
    for (uint index = partition_offsets[leaf] + tile * 256 + tid;
         index < partition_offsets[leaf + 1]; index += p.tiles * 256) {
        const uint row = row_indices[index];
        const float hessian = p.leaf_method == 1 ? 0.0f : curvature[row];
        QuerywiseAccumulate(float4(gradients[row], hessian, original_weights[row], 0), high, low);
    }
    high_scratch[tid] = high; low_scratch[tid] = low;
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) {
        const uint cell = 2 * (leaf * p.tiles + tile);
        partials[cell] = high_scratch[0]; partials[cell + 1] = low_scratch[0];
    }
}

// Bounded full-256-thread reduction, with at most 4096 dispatched groups.
// Output is UNNORMALIZED float2(sum loss,sum metric denominator), one per
// dispatched group. Host combines these bounded partials in double precision.
kernel void ReduceQuerywiseObjective(const device float2* query_stats [[buffer(0)]],
                                     device float2* partials [[buffer(1)]],
                                     constant QuerywiseParams& p [[buffer(2)]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint group [[threadgroup_position_in_grid]],
                                     uint groups [[threadgroups_per_grid]]) {
    threadgroup float4 high_scratch[256], low_scratch[256];
    float4 high = float4(0), low = float4(0);
    for (uint query = group * 256 + tid; query < p.groups; query += groups * 256)
        QuerywiseAccumulate(float4(query_stats[query], 0, 0), high, low);
    high_scratch[tid] = high; low_scratch[tid] = low;
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) partials[group] = high_scratch[0].xy + low_scratch[0].xy;
}
)METAL";
