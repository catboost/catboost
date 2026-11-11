#pragma once

// Appended to CBMMetalSource after KernelParams and the common MSL helpers.
// All reductions below require exactly 256 threads per threadgroup.
static const char* CBMMetalObjectiveSource = R"METAL(

inline float ObjectiveInvalidValue() {
    return as_type<float>(0x7fc00000u);
}

// Validate each row before reduction; negative curvatures must not cancel
// against positive rows and become an apparently valid leaf statistic.
inline float3 ObjectiveCustomValueDerivatives(float raw, float target, float weight) {
    if (weight == 0.0f) return float3(0.0f);
    if (!isfinite(raw) || !isfinite(target) || !isfinite(weight) || weight < 0.0f)
        return float3(ObjectiveInvalidValue());
#ifdef CBM_HAS_CUSTOM_OBJECTIVE
    const float3 result = CBMUserObjectiveValueDerivatives(raw, target, weight);
    if (all(isfinite(result)) && result.z >= 0.0f) return result;
#endif
    return float3(ObjectiveInvalidValue());
}

// Error-free TwoSum followed by renormalization of a two-float expansion.
// Both parts must survive every reduction level: a final plain float tree
// would otherwise lose the 1 in permutations of {2^24, 1, -2^24}.
inline void ObjectiveAddExpansion(thread float3& high, thread float3& low,
                                  float3 value) {
    const float3 sum = high + value;
    const float3 value_part = sum - high;
    const float3 error = (high - (sum - value_part)) + (value - value_part);
    const float3 tail = low + error;
    const float3 next = sum + tail;
    const float3 tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

inline void ObjectiveMergeExpansion(thread float3& high, thread float3& low,
                                    float3 other_high, float3 other_low) {
    ObjectiveAddExpansion(high, low, other_high);
    ObjectiveAddExpansion(high, low, other_low);
}

inline void ObjectiveAddExpansion(thread float& high, thread float& low,
                                  float value) {
    const float sum = high + value;
    const float value_part = sum - high;
    const float error = (high - (sum - value_part)) + (value - value_part);
    const float tail = low + error;
    const float next = sum + tail;
    const float tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

// Shared with the runtime's partition-statistics collector. Call uniformly
// from all 256 threads after storing each thread's high/low scratch pair.
inline void ObjectiveReduceExpansions(threadgroup float4* high_scratch,
                                      threadgroup float4* low_scratch, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 128; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float3 high = high_scratch[tid].xyz;
            float3 low = low_scratch[tid].xyz;
            ObjectiveMergeExpansion(high, low,
                high_scratch[tid + offset].xyz, low_scratch[tid + offset].xyz);
            high_scratch[tid] = float4(high, 0.0f);
            low_scratch[tid] = float4(low, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// targets/kernel/pointwise_targets.cu::CrossEntropyImpl. The sign-dependent
// exponential avoids overflow while retaining CUDA's float probability bounds.
// Logloss labels are binarized on the host; CrossEntropy retains soft labels.
inline float ObjectiveProbability(float raw) {
    const float exp_value = exp(-abs(raw));
    const float probability = raw >= 0.0f
        ? 1.0f / (1.0f + exp_value) : exp_value / (1.0f + exp_value);
    return clamp(probability, 1e-40f, 1.0f - 1e-40f);
}

// Metal's standard math library has no log1p. Correct the rounding error in
// 1+x so small positive losses survive when that addition rounds to exactly 1.
inline float ObjectiveLogOnePlus(float value) {
    const float sum = 1.0f + value;
    return sum == 1.0f ? value : log(sum) - ((sum - 1.0f) - value) / sum;
}

// targets/kernel/pointwise_targets.cu emits negative first derivatives and
// nonnegative second derivatives. A zero-weight observation contributes exactly
// zero, including when its unweighted arithmetic would overflow.
inline float2 ObjectiveGradientAndHessian(float target, float weight,
                                         float raw, uint objective,
                                         float objective_param) {
    if (weight == 0.0f) return float2(0.0f);
    if (!isfinite(raw) || !isfinite(target) || !isfinite(weight) || weight < 0.0f) {
        return float2(ObjectiveInvalidValue());
    }
#ifdef CBM_HAS_CUSTOM_OBJECTIVE
    if (objective == 20) return ObjectiveCustomValueDerivatives(raw, target, weight).yz;
#endif
    if (objective == 0) {
        return float2(weight * (target - raw), weight);
    }
    if (objective == 1 || objective == 2) {
        const float probability = ObjectiveProbability(raw);
        return float2(weight * (target - probability),
                      weight * probability * (1.0f - probability));
    }
    if (objective == 3) {
        // TPoissonTarget operates on raw log means. Do not clamp exp(raw):
        // overflow must reach the runtime's finite-output checks.
        const float mean = exp(raw);
        return float2(weight * (target - mean), weight * mean);
    }
    const float residual = target - raw;
    if (objective == 4) {
        // THuberTarget has ZERO Hessian at both exact +/-delta boundaries.
        const float delta = objective_param;
        return float2(weight * clamp(residual, -delta, delta),
                      abs(residual) < delta ? weight : 0.0f);
    }
    if (objective == 5) {
        // TExpectileTarget selects the (1-alpha) branch at residual == 0.
        const float multiplier = residual > 0.0f
            ? objective_param : 1.0f - objective_param;
        if (multiplier == 0.0f) return float2(0.0f);
        const float scale = 2.0f * multiplier;
        return float2(weight * (scale * residual), weight * scale);
    }
    if (objective >= 6) return weight * AdditionalObjectiveValueDerivatives(target, raw, objective, objective_param).yz;
    return float2(ObjectiveInvalidValue());
}

kernel void InitializeObjectivePredictions(device float* predictions [[buffer(0)]],
                                           constant KernelParams& p [[buffer(1)]],
                                           uint row [[thread_position_in_grid]]) {
    if (row < p.rows) predictions[row] = p.bias;
}

// Tree-search gradients and Hessians use the existing ensemble prediction.
// Starting the next tree also resets its oblivious-tree leaf bits.
kernel void ObjectiveDerivatives(const device float* targets [[buffer(0)]],
                                 const device float* sample_weights [[buffer(1)]],
                                 const device float* predictions [[buffer(2)]],
                                 device float* gradients [[buffer(3)]],
                                 device float* hessians [[buffer(4)]],
                                 device uint* leaf_ids [[buffer(5)]],
                                 constant KernelParams& p [[buffer(6)]],
                                 uint row [[thread_position_in_grid]]) {
    if (row < p.rows) {
        const float2 derivatives = ObjectiveGradientAndHessian(
            targets[row], sample_weights[row], predictions[row], p.objective,
            p.objective_param);
        gradients[row] = derivatives.x;
        hessians[row] = derivatives.y;
        leaf_ids[row] = 0;
    }
}

kernel void InitializeLeafValues(device float* raw_values [[buffer(0)]],
                                 device float* output_weights [[buffer(1)]],
                                 constant KernelParams& p [[buffer(2)]],
                                 uint leaf [[thread_position_in_grid]]) {
    if (leaf < p.leaves) {
        raw_values[leaf] = 0.0f;
        output_weights[leaf] = 0.0f;
    }
}

// Histogram and leaf estimation share a bounded per-leaf tile count. The
// runtime sets reserved0 to that count and allocates 2*leaves*reserved0 float4s.
inline uint ObjectiveLeafTileCount(constant KernelParams& p) {
    return max(p.reserved0, 1u);
}

// CUDA projects the current objective derivatives into each leaf after adding
// the current unshrunk leaf point to the ensemble cursor. Stable partition
// indices preserve the association of targets, weights and predictions.
// Dispatch 2D threadgroups (tiles_per_leaf, leaves), each with 256 threads.
// Partial layout: float4[2 * (leaf * tiles_per_leaf + tile)] is the high part
// of (grad,hess,weight,0), followed by a float4 containing its low part.
kernel void ReduceLeafObjectivePartials(
    const device float* targets [[buffer(0)]],
    const device float* sample_weights [[buffer(1)]],
    const device float* predictions [[buffer(2)]],
    const device float* raw_values [[buffer(3)]],
    const device uint* row_indices [[buffer(4)]],
    const device uint* partition_offsets [[buffer(5)]],
    device float4* partials [[buffer(6)]],
    constant KernelParams& p [[buffer(7)]],
    uint2 local_position [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float4 high_scratch[256];
    threadgroup float4 low_scratch[256];
    const uint tid = local_position.x;
    const uint leaf = group.y;
    const uint tile = group.x;
    const uint tile_count = ObjectiveLeafTileCount(p);
    if (leaf >= p.leaves || tile >= tile_count) return;

    const uint begin = partition_offsets[leaf];
    const uint end = partition_offsets[leaf + 1];
    const float leaf_value = raw_values[leaf];
    float3 high = float3(0.0f);
    float3 low = float3(0.0f);
    for (uint index = begin + tile * 256 + tid; index < end;
         index += tile_count * 256) {
        const uint row = row_indices[index];
        const float weight = sample_weights[row];
        const float2 derivatives = ObjectiveGradientAndHessian(
            targets[row], weight, predictions[row] + leaf_value, p.objective,
            p.objective_param);
        // CUDA widens these reductions; retain a float expansion because
        // Metal has no native float64. Preserve its low part across tiles.
        ObjectiveAddExpansion(high, low, float3(derivatives, weight));
    }
    high_scratch[tid] = float4(high, 0.0f);
    low_scratch[tid] = float4(low, 0.0f);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    if (tid == 0) {
        const uint partial = 2 * (leaf * tile_count + tile);
        partials[partial] = high_scratch[0];
        partials[partial + 1] = low_scratch[0];
    }
}

// oblivious_tree_leaves_estimator.cpp::WriteSecondDerivatives adds lambda;
// descent_helpers.cpp::UpdateMoveDirectionDiagonal uses g/(h+lambda+1e-20)
// only for a positive diagonal. Gradient estimation substitutes original sample
// weight for h, as WriteSecondDerivatives does. RegularizeImpl zeros empty
// leaves, not zero-Hessian leaves. With No backtracking each step has size 1.
// AddRidgeToTargetFunction is false: no extra -lambda*leaf_value gradient.
// Dispatch one threadgroup of 256 threads per leaf after the partial pass.
kernel void EstimateNewtonLeafValues(const device float4* partials [[buffer(0)]],
                                     device float* raw_values [[buffer(1)]],
                                     device float* output_weights [[buffer(2)]],
                                     constant KernelParams& p [[buffer(3)]],
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
        const float3 statistics = high_scratch[0].xyz + low_scratch[0].xyz;
        const float gradient = statistics.x;
        const float hessian = statistics.y;
        const float weight = statistics.z;
        output_weights[leaf] = weight;
        const float diagonal = (p.leaf_method == 1 ? weight : hessian) + p.l2;
        // QuerySoftMax permits signed curvature parameters. Preserve its
        // literal Hessian when the regularized Newton diagonal is positive.
        // Combination's negated YetiRank coefficient can also produce a
        // nonpositive diagonal; CUDA leaves its move direction at zero.
        const bool grouped = p.objective == 12 || p.objective == 13 || p.objective == 19;
        if (!all(isfinite(statistics)) || (!grouped && hessian < 0.0f) || weight < 0.0f
            || (grouped && p.objective != 19 && p.leaf_method == 0 && weight >= 1e-20f && diagonal <= 0.0f)
            || !isfinite(diagonal) || !isfinite(raw_values[leaf])) {
            // Preserve a detectable error for the runtime's finite-output
            // checks, instead of silently selecting a finite replacement.
            raw_values[leaf] = ObjectiveInvalidValue();
        } else if (weight < 1e-20f) {
            raw_values[leaf] = 0.0f;
        } else if (diagonal > 0.0f) {
            const float next = raw_values[leaf] + gradient / (diagonal + 1e-20f);
            raw_values[leaf] = isfinite(next) ? next : ObjectiveInvalidValue();
        }
    }
}

// doc_parallel_boosting.h rescales the completed tree, after leaf estimation.
kernel void FinalizeLeafValues(const device float* raw_values [[buffer(0)]],
                               device float* values [[buffer(1)]],
                               constant KernelParams& p [[buffer(2)]],
                               uint leaf [[thread_position_in_grid]]) {
    if (leaf < p.leaves) values[leaf] = raw_values[leaf] * p.learning_rate;
}

kernel void AddObjectiveBinModelValue(const device uint* leaf_ids [[buffer(0)]],
                                     const device float* values [[buffer(1)]],
                                     device float* predictions [[buffer(2)]],
                                     constant KernelParams& p [[buffer(3)]],
                                     uint row [[thread_position_in_grid]]) {
    if (row < p.rows) predictions[row] += values[leaf_ids[row]];
}

// CUDA reports the negative objective sum. Public metrics reverse that sign
// and divide by total sample weight; the runtime takes sqrt for RMSE only.
// Poisson's exp(raw)-target*raw can be negative and is not shifted or clamped.
// softplus(raw)-target*raw is rearranged to avoid exp overflow and catastrophic
// cancellation for correctly classified large logits. Inputs are raw logits.
kernel void ReduceObjectiveLoss(const device float* targets [[buffer(0)]],
                                const device float* sample_weights [[buffer(1)]],
                                const device float* predictions [[buffer(2)]],
                                device float* partials [[buffer(3)]],
                                constant KernelParams& p [[buffer(4)]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]],
                                uint groups [[threadgroups_per_grid]]) {
    threadgroup float2 scratch[256];
    float high = 0.0f;
    float low = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float weight = sample_weights[row];
        if (weight == 0.0f) continue;
        const float target = targets[row];
        const float raw = predictions[row];
        float loss;
        if (!isfinite(raw) || !isfinite(target) || !isfinite(weight)
            || weight < 0.0f || !isfinite(p.total_weight) || p.total_weight <= 0.0f) {
            loss = ObjectiveInvalidValue();
        }
#ifdef CBM_HAS_CUSTOM_OBJECTIVE
        else if (p.objective == 20) {
            loss = -ObjectiveCustomValueDerivatives(raw, target, weight).x / p.total_weight;
        }
#endif
        else if (p.objective == 0) {
            const float residual = target - raw;
            const float weighted_residual = residual * sqrt(weight / p.total_weight);
            loss = weighted_residual * weighted_residual;
        } else if (p.objective == 1 || p.objective == 2) {
            const float linear_term = raw >= 0.0f
                ? (1.0f - target) * raw : -target * raw;
            loss = (weight / p.total_weight)
                * (linear_term + ObjectiveLogOnePlus(exp(-abs(raw))));
        } else if (p.objective == 3) {
            loss = (weight / p.total_weight) * (exp(raw) - target * raw);
        } else if (p.objective == 4) {
            const float mismatch = abs(target - raw);
            const float delta = p.objective_param;
            if (mismatch < delta) {
                const float weighted_mismatch = mismatch
                    * sqrt(0.5f * (weight / p.total_weight));
                loss = weighted_mismatch * weighted_mismatch;
            } else {
                loss = ((weight / p.total_weight) * delta)
                    * (mismatch - 0.5f * delta);
            }
        } else if (p.objective == 5) {
            const float residual = target - raw;
            const float multiplier = residual > 0.0f
                ? p.objective_param : 1.0f - p.objective_param;
            const float weighted_residual = multiplier == 0.0f ? 0.0f
                : residual * sqrt((weight / p.total_weight) * multiplier);
            loss = weighted_residual * weighted_residual;
        } else if (p.objective >= 6) {
            loss = (weight / p.total_weight) * AdditionalObjectiveValueDerivatives(target, raw, p.objective, p.objective_param).x;
            if (p.objective == 10) loss *= 2.0f; // Report MAE; training uses half absolute loss.
        } else {
            loss = ObjectiveInvalidValue();
        }
        ObjectiveAddExpansion(high, low, loss);
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
    // Only the final group result is rounded; the host adds groups in double.
    if (tid == 0) partials[group] = scratch[0].x + scratch[0].y;
}
)METAL";
