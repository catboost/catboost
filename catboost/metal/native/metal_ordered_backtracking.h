#pragma once

// Append after shared objective/backtracking and Ordered session sources.
// CUDA's dynamic estimator has ONE walker over all prefix/full tasks, so the
// host sums these task objectives and uses one trial step for every task.
static const char* CBMMetalOrderedBacktrackingSource = R"METAL(
kernel void OrderedBacktrackingDirections(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    const device uint4* tasks [[buffer(5)]], const device float* values [[buffer(6)]],
    device float* directions [[buffer(7)]], device float* leaf_weights [[buffer(8)]],
    device float2* direction_dot [[buffer(9)]], device float* task_mass [[buffer(10)]],
    device atomic_uint* status [[buffer(11)]], constant OrderedTrainingParams& p [[buffer(12)]],
    constant OrderedStepParams& step [[buffer(13)]], uint2 local [[thread_position_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]]) {
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
        const float3 sh = high_scratch[0].xyz, sl = low_scratch[0].xyz;
        const float mass = mass_scratch[0].x + mass_scratch[0].y;
        if (!leaf) task_mass[task_id] = mass;
        leaf_weights[output] = sh.z + sl.z;
        float gh = sh.x, gl = sl.x, dh = sh.y, dl = sl.y;
        if (p.normalize) {
            const float2 g = mass > 0 ? BacktrackingDivideExpansion(gh, gl, mass) : float2(0);
            const float2 h = mass > 0 ? BacktrackingDivideExpansion(dh, dl, mass) : float2(0);
            gh = g.x; gl = g.y; dh = h.x; dl = h.y;
        }
        ObjectiveAddExpansion(dh, dl, p.l2);
        const float diagonal = dh + dl;
        float direction = 0;
        if (diagonal > 0) {
            ObjectiveAddExpansion(dh, dl, 1e-20f);
            direction = BacktrackingDirection(gh, gl, dh, dl);
        }
        directions[output] = direction;
        float dot_high = gh * direction, dot_low = fma(gh, direction, -dot_high);
        ObjectiveAddExpansion(dot_high, dot_low, gl * direction);
        direction_dot[output] = float2(dot_high, dot_low);
        if (!all(isfinite(sh + sl)) || !isfinite(mass) || !isfinite(diagonal) || !isfinite(direction) ||
            !isfinite(dot_high) || !isfinite(dot_low) || sh.y + sl.y < 0 || sh.z + sl.z < 0)
            atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void OrderedBacktrackingCandidate(const device float* current [[buffer(0)]],
    const device float* directions [[buffer(1)]], const device float* weights [[buffer(2)]],
    device float* candidate [[buffer(3)]], constant OrderedTrainingParams& p [[buffer(4)]],
    constant OrderedStepParams& step [[buffer(5)]], constant BacktrackingParams& b [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index < step.tasks * step.leaves) {
        const uint output = (index / step.leaves) * (1u << p.depth) + index % step.leaves;
        candidate[output] = weights[output] < 1e-20f ? 0.0f : fma(b.step, directions[output], current[output]);
    }
}

// Shared BacktrackingRowScore takes constant KernelParams; this overload
// accepts Ordered's own ABI and preserves exactly the same objective scale.
inline float OrderedBacktrackingScore(float target, float raw, constant OrderedTrainingParams& p) {
    if (!isfinite(raw) || !isfinite(target)) return ObjectiveInvalidValue();
    if (p.objective == 0) return (target - raw) * (target - raw);
    if (p.objective == 1 || p.objective == 2)
        return (raw >= 0 ? (1.0f - target) * raw : -target * raw) + ObjectiveLogOnePlus(exp(-abs(raw)));
    if (p.objective == 3) return exp(raw) - target * raw;
    const float residual = target - raw;
    if (p.objective == 4) {
        const float mismatch = abs(residual);
        return mismatch < p.objective_param ? .5f * mismatch * mismatch : p.objective_param * (mismatch - .5f * p.objective_param);
    }
    if (p.objective == 5) {
        const float multiplier = residual > 0 ? p.objective_param : 1.0f - p.objective_param;
        return multiplier == 0 ? 0 : (multiplier * residual) * residual;
    }
    return AdditionalObjectiveValueDerivatives(target, raw, p.objective, p.objective_param).x;
}

// Each task contributes its negative prefix objective (normalized by total
// prefix weight only when configured). The host sums tasks in double before
// one common AnyImprovement/Armijo acceptance decision. Preserve nonfinite
// trial scores so rejection can recover by halving, without poisoning status.
kernel void OrderedBacktrackingObjective(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    const device uint4* tasks [[buffer(5)]], const device float* values [[buffer(6)]],
    const device float* task_mass [[buffer(7)]], device float2* loss [[buffer(8)]],
    constant OrderedTrainingParams& p [[buffer(9)]], constant OrderedStepParams& step [[buffer(10)]],
    uint tid [[thread_position_in_threadgroup]], uint task_id [[threadgroup_position_in_grid]]) {
    if (task_id >= step.tasks) return;
    const uint4 task = tasks[task_id];
    threadgroup float2 scratch[256];
    float high = 0, low = 0;
    for (uint position = tid; position < task.x; position += 256) {
        const uint row = permutations[task.w * p.rows + position];
        const float weight = weights[row];
        if (weight == 0) continue;
        const float raw = cursors[task.z + position] + values[task_id * (1u << p.depth) + leaf_ids[task.w * p.reserved0 + row]];
        ObjectiveAddExpansion(high, low, -weight * OrderedBacktrackingScore(targets[row], raw, p));
    }
    if (p.normalize) {
        const float2 normalized = task_mass[task_id] > 0 ? BacktrackingDivideExpansion(high, low, task_mass[task_id]) : float2(0);
        high = normalized.x; low = normalized.y;
    }
    scratch[tid] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            high = scratch[tid].x; low = scratch[tid].y;
            ObjectiveAddExpansion(high, low, scratch[tid + stride].x);
            ObjectiveAddExpansion(high, low, scratch[tid + stride].y);
            scratch[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) loss[task_id] = scratch[0];
}
)METAL";
