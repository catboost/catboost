#pragma once
// Source: cuda/targets/kernel/multilogit.cu:215-350 and 492-607.
// Both Hessians are diagonal even though CUDA submits them to its symmetric
// block solver. Uncertainty's first dimension uses the natural gradient,
// deliberately omitting inverse variance from its mean gradient/Hessian.
static const char* CBMMetalMultioutputMathSource = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct MultioutputParams {
    uint rows, dimensions, objective, leaves;
    float l2, min_leaf_weight;
    uint leaf_method, reserved;
};
inline void MultioutputAdd(thread float& high, thread float& low, float value) {
    const float sum = high + value;
    const float part = sum - high;
    const float error = (high - (sum - part)) + (value - part);
    const float tail = low + error;
    const float next = sum + tail, tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}
// Buffer bindings intentionally match the multiclass objective dispatch.
kernel void MultioutputDerivatives(
    const device float* targets [[buffer(0)]], const device float* weights [[buffer(1)]],
    const device float* predictions [[buffer(2)]], device float* gradients [[buffer(3)]],
    device float* hessian [[buffer(4)]], device float* losses [[buffer(5)]],
    constant MultioutputParams& p [[buffer(6)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const float w = weights[row];
    if (w == 0.0f) {
        for (uint d = 0; d < p.dimensions; ++d) {
            gradients[d * p.rows + row] = 0.0f;
            hessian[d * p.rows + row] = 0.0f;
        }
        losses[row] = 0.0f;
        return;
    }
    if (p.objective == 2) {
        float loss = 0.0f, tail = 0.0f;
        for (uint d = 0; d < p.dimensions; ++d) {
            const uint i = d * p.rows + row;
            const float error = targets[i] - predictions[i];
            gradients[i] = error * w;
            hessian[i] = w;
            MultioutputAdd(loss, tail, error * error * w);
        }
        losses[row] = loss + tail;
    } else if (p.objective == 3) {
        const float error = targets[row] - predictions[row];
        const float log_sigma = predictions[p.rows + row];
        const float inverse_variance = exp(min(-2.0f * log_sigma, 70.0f));
        const float normalized_error = error * error * inverse_variance;
        gradients[row] = w * error;
        gradients[p.rows + row] = w * (normalized_error - 1.0f);
        hessian[row] = w;
        hessian[p.rows + row] = 2.0f * w * error * error * inverse_variance;
        losses[row] = w * (0.9189385332046f + log_sigma + 0.5f * normalized_error);
    } else {
        float loss = 0.0f, tail = 0.0f;
        for (uint d = 0; d < p.dimensions; ++d) {
            const uint i = d * p.rows + row;
            const float value = predictions[i], target = targets[i];
            const float exponential = exp(-abs(value));
            // Unlike OneVsAll, CUDA MultiCrossEntropy does not ClipProb.
            const float probability = value >= 0.0f ? 1.0f / (1.0f + exponential)
                : exponential / (1.0f + exponential);
            gradients[i] = (target - probability) * w;
            hessian[i] = probability * (1.0f - probability) * w;
            MultioutputAdd(loss, tail, (max(value, 0.0f) - target * value) + log(1.0f + exponential));
        }
        losses[row] = w * ((loss + tail) / float(p.dimensions));
    }
}
// One 256-thread group per leaf/statistic. Statistics: [weight,G[D],Hdiag[D]].
kernel void MultioutputReduceLeafStats(
    const device float* gradients [[buffer(0)]], const device float* hessian [[buffer(1)]],
    const device float* weights [[buffer(2)]], const device uint* row_indices [[buffer(3)]],
    const device uint* offsets [[buffer(4)]], device float* statistics [[buffer(5)]],
    constant MultioutputParams& p [[buffer(6)]], uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint width = 1 + 2 * p.dimensions;
    const uint leaf = group / width, component = group % width;
    if (leaf >= p.leaves) return;
    float high = 0.0f, low = 0.0f;
    for (uint pos = offsets[leaf] + tid; pos < offsets[leaf + 1]; pos += 256) {
        const uint row = row_indices[pos];
        const float value = component == 0 ? weights[row] : component <= p.dimensions
            ? gradients[(component - 1) * p.rows + row]
            : hessian[(component - 1 - p.dimensions) * p.rows + row];
        MultioutputAdd(high, low, value);
    }
    threadgroup float2 partial[256];
    partial[tid] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = partial[tid].x; low = partial[tid].y;
            MultioutputAdd(high, low, partial[tid + stride].x);
            MultioutputAdd(high, low, partial[tid + stride].y);
            partial[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) statistics[leaf * width + component] = partial[0].x + partial[0].y;
}
// Workspace binding is retained for interchange with the coupled solver.
kernel void MultioutputSolveLeaves(
    const device float* statistics [[buffer(0)]], device float* workspace [[buffer(1)]],
    device float* directions [[buffer(2)]], device uint* status [[buffer(3)]],
    constant MultioutputParams& p [[buffer(4)]], uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const device float* stats = statistics + leaf * (1 + 2 * p.dimensions);
    const float weight = stats[0];
    status[leaf] = 0;
    for (uint d = 0; d < p.dimensions; ++d) directions[leaf * p.dimensions + d] = 0.0f;
    if (!isfinite(weight) || weight < 0.0f) { status[leaf] = 1; return; }
    if (weight < p.min_leaf_weight || weight == 0.0f) return;
    for (uint d = 0; d < p.dimensions; ++d) {
        const float gradient = stats[1 + d];
        const float curvature = p.leaf_method == 1 ? weight : stats[1 + p.dimensions + d];
        const float denominator = curvature + p.l2;
        if (!isfinite(gradient) || !isfinite(denominator) || curvature < 0.0f) {
            status[leaf] = 1; return;
        }
        // A nonempty singular Newton block fails CUDA Cholesky as well.
        if (denominator <= 0.0f) { status[leaf] = 2; return; }
        const float result = gradient / denominator;
        if (!isfinite(result)) { status[leaf] = 1; return; }
        directions[leaf * p.dimensions + d] = result;
    }
}
// Same vector walker and bindings as multiclass, with float matrix targets.
kernel void MultioutputBacktrackingReduceObjective(
    const device float* targets [[buffer(0)]], const device float* weights [[buffer(1)]],
    const device float* base [[buffer(2)]], const device uint* leaf_ids [[buffer(3)]],
    const device float* trial_values [[buffer(4)]], device float2* partials [[buffer(5)]],
    constant MultioutputParams& p [[buffer(6)]], uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]]) {
    float high = 0.0f, low = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        if (weights[row] == 0.0f) continue;
        const device float* point = trial_values + leaf_ids[row] * p.dimensions;
        float loss = 0.0f, tail = 0.0f;
        if (p.objective == 3) {
            const float mean = base[row] + point[0];
            const float log_sigma = base[p.rows + row] + point[1];
            const float error = targets[row] - mean;
            loss = 0.9189385332046f + log_sigma
                + 0.5f * exp(min(-2.0f * log_sigma, 70.0f)) * error * error;
        } else {
            for (uint d = 0; d < p.dimensions; ++d) {
                const float value = base[d * p.rows + row] + point[d];
                const float target = targets[d * p.rows + row];
                const float error = target - value;
                const float term = p.objective == 2 ? error * error
                    : (max(value, 0.0f) - target * value) + log(1.0f + exp(-abs(value)));
                MultioutputAdd(loss, tail, term);
            }
            loss += tail;
            if (p.objective >= 4) loss /= float(p.dimensions);
        }
        MultioutputAdd(high, low, -weights[row] * loss);
    }
    threadgroup float2 scratch[256];
    scratch[tid] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = scratch[tid].x; low = scratch[tid].y;
            MultioutputAdd(high, low, scratch[tid + stride].x);
            MultioutputAdd(high, low, scratch[tid + stride].y);
            scratch[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[group] = scratch[0];
}
)METAL";
