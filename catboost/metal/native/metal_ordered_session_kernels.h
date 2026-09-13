#pragma once

// Concatenate after shared scalar objective and Ordered foundation sources.
static const char* CBMMetalOrderedSessionSource = R"METAL(
struct OrderedTrainingParams {
    uint rows, features, candidates, iterations;
    uint depth, objective, score_function, leaf_method;
    uint leaf_iterations, permutations, min_fold_size, normalize;
    float learning_rate, l2, bias, fold_growth;
    float objective_param;
    uint reserved0, reserved1, reserved2;
};
struct OrderedStepParams { uint tasks, leaves, cursor_count, selected_permutation; };

kernel void OrderedSessionGatherExact(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    device float* compact_targets [[buffer(5)]], device float* compact_weights [[buffer(6)]],
    device float* compact_predictions [[buffer(7)]], device uint* compact_leaf_ids [[buffer(8)]],
    constant OrderedTrainingParams& p [[buffer(9)]], constant uint4& task [[buffer(10)]],
    uint position [[thread_position_in_grid]]) {
    if (position < task.x) {
        const uint row = permutations[task.w * p.rows + position];
        compact_targets[position] = targets[row]; compact_weights[position] = weights[row];
        compact_predictions[position] = cursors[task.z + position]; compact_leaf_ids[position] = leaf_ids[task.w * p.reserved0 + row];
    }
}

// Preserve feature-parallel CUDA's actual MVS call-site argument:
// BootstrappedWeights(random, &target.Weights), not WeightedTarget.
kernel void OrderedSessionMvsInput(const device float2* derivatives [[buffer(0)]],
    const device uint4* folds [[buffer(1)]], device float* magnitudes [[buffer(2)]],
    constant BootstrapParams& p [[buffer(3)]], uint occurrence [[thread_position_in_grid]]) {
    if (occurrence < p.rows) magnitudes[occurrence] = derivatives[folds[0].z + occurrence].y;
}

kernel void OrderedSessionFeatureNoise(const device float* noise [[buffer(0)]],
    device float4* feature_options [[buffer(1)]], constant OrderedTrainingParams& p [[buffer(2)]],
    uint feature [[thread_position_in_grid]]) {
    if (feature < p.features) feature_options[feature].z = noise[feature];
}

kernel void OrderedSessionDerivatives(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint4* tasks [[buffer(4)]],
    device float2* derivatives [[buffer(5)]], device atomic_uint* status [[buffer(6)]],
    constant OrderedTrainingParams& p [[buffer(7)]], constant OrderedStepParams& step [[buffer(8)]],
    uint2 position [[thread_position_in_grid]]) {
    if (position.y >= step.tasks) return;
    const uint4 task = tasks[position.y];
    if (position.x >= task.y) return;
    const uint row = permutations[task.w * p.rows + position.x], slot = task.z + position.x;
    const float2 d = ObjectiveGradientAndHessian(targets[row], weights[row], cursors[slot], p.objective, p.objective_param);
    derivatives[slot] = float2(d.x, p.score_function ? d.y : weights[row]);
    if (!all(isfinite(d)) || d.y < 0.0f) atomic_store_explicit(status, 1u, memory_order_relaxed);
}

kernel void OrderedSessionFindWinner(const device float2* scores [[buffer(0)]],
    const device uint2* candidates [[buffer(1)]], device SplitState* output [[buffer(2)]],
    constant OrderedTrainingParams& p [[buffer(3)]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float values[256];
    threadgroup uint indices[256], invalid[256];
    float best = FLT_MAX;
    uint index = 0xffffffffu, bad = 0;
    for (uint candidate = tid; candidate < p.candidates; candidate += 256) {
        const float score = scores[candidate].y;
        if (!isfinite(score)) bad = 1;
        else if (score < best || (score == best && candidate < index)) { best = score; index = candidate; }
    }
    values[tid] = best; indices[tid] = index; invalid[tid] = bad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const uint other = tid + stride;
            if (values[other] < values[tid] || (values[other] == values[tid] && indices[other] < indices[tid])) {
                values[tid] = values[other]; indices[tid] = indices[other];
            }
            invalid[tid] |= invalid[other];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        const uint winner = indices[0];
        SplitState result = {};
        result.index = winner;
        result.valid = winner != 0xffffffffu && scores[winner].x < FLT_MAX;
        result.reserved0 = invalid[0];
        if (result.valid) {
            result.feature = candidates[winner].x; result.bin = candidates[winner].y & 0xffu;
            result.type = candidates[winner].y >> 31;
            result.score = scores[winner].x; result.gain = scores[winner].y;
        }
        output[0] = result;
    }
}

kernel void OrderedSessionUpdateLeafIds(const device uchar* bins [[buffer(0)]],
    const device SplitState* winner [[buffer(1)]], device uint* leaf_ids [[buffer(2)]],
    constant OrderedTrainingParams& p [[buffer(3)]], constant OrderedStepParams& step [[buffer(4)]],
    uint2 position [[thread_position_in_grid]]) {
    const uint row = position.x, bank = position.y;
    if (row < p.rows && bank < (p.reserved0 ? p.permutations : 1u)) {
        const SplitState split = winner[0];
        const uint value = bins[(bank * p.rows + row) * p.features + split.feature];
        leaf_ids[bank * p.rows + row] |= uint(split.type ? value == split.bin : value > split.bin) * step.leaves;
    }
}

// One 256-thread group per task and leaf. All objective math uses the shared
// CUDA-derived ObjectiveGradientAndHessian helper. Every task retains a
// distinct baseline cursor and unshrunk leaf point throughout leaf iterations.
kernel void OrderedSessionEstimateLeaves(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    const device uint4* tasks [[buffer(5)]], device float* values [[buffer(6)]],
    device float* leaf_weights [[buffer(7)]], device atomic_uint* status [[buffer(8)]],
    constant OrderedTrainingParams& p [[buffer(9)]], constant OrderedStepParams& step [[buffer(10)]],
    uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint leaf = group.x, task_id = group.y, tid = local.x;
    if (leaf >= step.leaves || task_id >= step.tasks) return;
    const uint4 task = tasks[task_id];
    const uint output = task_id * (1u << p.depth) + leaf;
    const float point = values[output];
    threadgroup float4 high_scratch[256], low_scratch[256];
    threadgroup float2 mass_scratch[256];
    float3 high = 0.0f, low = 0.0f;
    float mass_high = 0.0f, mass_low = 0.0f;
    for (uint position = tid; position < task.x; position += 256) {
        const uint row = permutations[task.w * p.rows + position];
        const float weight = weights[row];
        ObjectiveAddExpansion(mass_high, mass_low, weight);
        if (leaf_ids[task.w * p.reserved0 + row] != leaf) continue;
        const float2 d = ObjectiveGradientAndHessian(targets[row], weight,
            cursors[task.z + position] + point, p.objective, p.objective_param);
        ObjectiveAddExpansion(high, low, float3(d.x, p.leaf_method ? weight : d.y, weight));
    }
    high_scratch[tid] = float4(high, 0); low_scratch[tid] = float4(low, 0);
    mass_scratch[tid] = float2(mass_high, mass_low);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            mass_high = mass_scratch[tid].x; mass_low = mass_scratch[tid].y;
            ObjectiveAddExpansion(mass_high, mass_low, mass_scratch[tid + stride].x);
            ObjectiveAddExpansion(mass_high, mass_low, mass_scratch[tid + stride].y);
            mass_scratch[tid] = float2(mass_high, mass_low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) {
        const float3 s = high_scratch[0].xyz + low_scratch[0].xyz;
        const float mass = mass_scratch[0].x + mass_scratch[0].y;
        const float diagonal = s.y + p.l2 * (p.normalize ? mass : 1.0f);
        float next = point;
        if (s.z < 1e-20f) next = 0.0f;
        else if (diagonal > 0.0f) next += s.x / (diagonal + 1e-20f);
        values[output] = next; leaf_weights[output] = s.z;
        if (!all(isfinite(s)) || !isfinite(mass) || !isfinite(diagonal) || !isfinite(next) || s.y < 0.0f || s.z < 0.0f)
            atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void OrderedSessionApplyValues(const device float* cursors [[buffer(0)]],
    const device uint* permutations [[buffer(1)]], const device uint* leaf_ids [[buffer(2)]],
    const device uint4* tasks [[buffer(3)]], const device float* values [[buffer(4)]],
    device float* updated [[buffer(5)]], device float* published [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]], constant OrderedTrainingParams& p [[buffer(8)]],
    constant OrderedStepParams& step [[buffer(9)]], uint2 position [[thread_position_in_grid]]) {
    if (position.y >= step.tasks) return;
    const uint4 task = tasks[position.y];
    if (position.x >= task.y) return;
    const uint row = permutations[task.w * p.rows + position.x], slot = task.z + position.x;
    const float value = values[position.y * (1u << p.depth) + leaf_ids[task.w * p.reserved0 + row]] * p.learning_rate;
    const float point = cursors[slot] + value;
    updated[slot] = point;
    if (position.y + 1 == step.tasks) published[row] = point;
    if (!isfinite(point)) atomic_store_explicit(status, 1u, memory_order_relaxed);
}

kernel void OrderedSessionPublish(const device float* cursors [[buffer(0)]],
    const device uint* permutations [[buffer(1)]], const device uint4* tasks [[buffer(2)]],
    device float* published [[buffer(3)]], constant OrderedTrainingParams& p [[buffer(4)]],
    constant OrderedStepParams& step [[buffer(5)]], uint position [[thread_position_in_grid]]) {
    if (position < p.rows) {
        const uint4 task = tasks[step.tasks - 1];
        published[permutations[task.w * p.rows + position]] = cursors[task.z + position];
    }
}
)METAL";
