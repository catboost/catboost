#pragma once

// Concatenate after CBMMetalBootstrapSource. Unlike scalar score noise, CUDA
// multiclass uses greedy_subsets_searcher::ComputeTargetVarianceImpl on the
// POST-bootstrap target. CUDA explicitly rejects multiclass MVS.
static const char* CBMMetalMulticlassBootstrapSource = R"METAL(
struct MulticlassBootstrapParams {
    uint rows, dimensions, multi_logit, reserved;
};

// One bootstrap draw per OBJECT is shared across every active class. Original
// observation weights remain available for final leaf estimation.
kernel void ApplyMulticlassBootstrap(device float* gradients [[buffer(0)]],
                                     const device float* original_weights [[buffer(1)]],
                                     const device float* multipliers [[buffer(2)]],
                                     device float* structure_weights [[buffer(3)]],
                                     constant MulticlassBootstrapParams& p [[buffer(4)]],
                                     uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.dimensions) return;
    const uint row = index % p.rows;
    const float multiplier = multipliers[row];
    gradients[index] = multiplier == 0.0f ? 0.0f : gradients[index] * multiplier;
    if (index < p.rows) {
        structure_weights[row] = multiplier == 0.0f ? 0.0f : original_weights[row] * multiplier;
    }
}

// CUDA greedy_subsets_searcher/kernel/compute_scores.cu computes
// sum_{i,k}(g_ik^2/w_i) / sum_i(w_i), with w_i>1e-15, AFTER bootstrap.
// MultiClass stores C-1 gradients: add (-sum(active gradients))^2/w_i.
// OneVsAll stores every gradient and does not reconstruct another dimension.
// No mean subtraction, no division by class count, no ZeroAwareDivide.
//
// Dispatch full groups of256. Output partial .x numerator/rows and .y
// weight/rows; host sums each column in double and takes sqrt(sumx/sumy).
// Scaling both quantities limits reduction overflow without changing ratio.
kernel void ReduceMulticlassScoreStatistics(const device float* gradients [[buffer(0)]],
                                            const device float* weights [[buffer(1)]],
                                            device float2* partials [[buffer(2)]],
                                            constant MulticlassBootstrapParams& p [[buffer(3)]],
                                            uint tid [[thread_position_in_threadgroup]],
                                            uint group [[threadgroup_position_in_grid]],
                                            uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 scratch[256];
    float2 sum = float2(0.0f), correction = float2(0.0f);
    const float root_rows = sqrt(float(p.rows));
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float weight = weights[row];
        if (weight <= 1e-15f) continue;
        const float root_weight = sqrt(weight);
        float energy = 0.0f, energy_correction = 0.0f;
        float class_sum = 0.0f;
        for (uint k = 0; k < p.dimensions; ++k) {
            const float gradient = gradients[k * p.rows + row];
            class_sum += gradient;
            const float normalized = (gradient / root_weight) / root_rows;
            const float value = normalized * normalized - energy_correction;
            const float next = energy + value;
            energy_correction = (next - energy) - value;
            energy = next;
        }
        if (p.multi_logit) {
            const float normalized = (class_sum / root_weight) / root_rows;
            energy += normalized * normalized;
        }
        const float2 value = float2(energy, weight / float(p.rows)) - correction;
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
)METAL";
