#pragma once

// Ordered Langevin projects clean normalized derivative statistics before
// adding source-host noise. Pack only active tasks, each with actualLeaves
// entries, because CUDA draws one vector whose 128-element RNG blocks may
// cross task boundaries. Native leaf storage still uses maxLeaves strides.
static const char* CBMMetalOrderedLangevinSource = R"METAL(
struct OrderedLangevinParams { uint active_tasks, query, trial, initial; };
kernel void OrderedLangevinStatistics(const device float* targets [[buffer(0)]],
    const device float* weights [[buffer(1)]], const device float* cursors [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    const device uint4* tasks [[buffer(5)]], const device float* values [[buffer(6)]],
    const device float* query_gradient [[buffer(7)]], const device float* query_hessian [[buffer(8)]],
    const device float* query_weights [[buffer(9)]], const device uint* active_tasks [[buffer(10)]],
    device float4* statistics [[buffer(11)]], device float* leaf_weights [[buffer(12)]],
    device float* task_mass [[buffer(13)]], device atomic_uint* status [[buffer(14)]],
    constant OrderedTrainingParams& p [[buffer(15)]], constant OrderedStepParams& step [[buffer(16)]],
    constant OrderedLangevinParams& l [[buffer(17)]],
    uint2 local [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint leaf = group.x, packed_task = group.y, tid = local.x;
    if (leaf >= step.leaves || packed_task >= l.active_tasks) return;
    const uint task_id = active_tasks[packed_task], output = task_id * (1u << p.depth) + leaf;
    const uint4 task = tasks[task_id];
    threadgroup float4 high_scratch[256], low_scratch[256];
    threadgroup float2 mass_scratch[256];
    float3 high = 0.0f, low = 0.0f;
    float mass_high = 0.0f, mass_low = 0.0f;
    for (uint position = tid; position < task.x; position += 256) {
        const uint row = permutations[task.w * p.rows + position], slot = task.z + position;
        const float weight = l.query ? query_weights[slot] : weights[row];
        ObjectiveAddExpansion(mass_high, mass_low, weight);
        if (leaf_ids[task.w * p.reserved0 + row] != leaf) continue;
        const float2 d = l.query ? float2(query_gradient[slot], query_hessian[slot]) :
            ObjectiveGradientAndHessian(targets[row], weight, cursors[slot] + values[output], p.objective, p.objective_param);
        ObjectiveAddExpansion(high, low, float3(d.x, p.leaf_method ? weight : d.y, weight));
    }
    high_scratch[tid] = float4(high, 0); low_scratch[tid] = float4(low, 0);
    mass_scratch[tid] = float2(mass_high, mass_low);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    OrderedQueryReduceExpansion(mass_scratch, tid);
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
        if (p.reserved1) {
            const float ridge = -p.l2 * values[output];
            ObjectiveAddExpansion(gh, gl, ridge);
            ObjectiveAddExpansion(gh, gl, fma(-p.l2, values[output], -ridge));
        }
        ObjectiveAddExpansion(dh, dl, p.l2);
        statistics[packed_task * step.leaves + leaf] = float4(gh, gl, dh, dl);
        if (!l.trial && (!all(isfinite(sh + sl)) || !isfinite(mass) || !all(isfinite(float4(gh, gl, dh, dl))) ||
                        sh.z + sl.z < 0)) atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

kernel void OrderedLangevinDirections(const device float4* statistics [[buffer(0)]],
    const device float2* gradient_noise [[buffer(1)]], const device float2* hessian_noise [[buffer(2)]],
    const device uint* active_tasks [[buffer(3)]], device float* directions [[buffer(4)]],
    device float2* direction_dot [[buffer(5)]], device atomic_uint* status [[buffer(6)]],
    constant OrderedTrainingParams& p [[buffer(7)]], constant OrderedStepParams& step [[buffer(8)]],
    constant OrderedLangevinParams& l [[buffer(9)]], uint index [[thread_position_in_grid]]) {
    if (index >= l.active_tasks * step.leaves) return;
    const uint output = active_tasks[index / step.leaves] * (1u << p.depth) + index % step.leaves;
    const float4 clean = statistics[index];
    float gh = clean.x, gl = clean.y, dh = clean.z, dl = clean.w;
    ObjectiveAddExpansion(gh, gl, gradient_noise[index].x);
    ObjectiveAddExpansion(gh, gl, gradient_noise[index].y);
    if (l.initial) {
        ObjectiveAddExpansion(dh, dl, hessian_noise[index].x);
        ObjectiveAddExpansion(dh, dl, hessian_noise[index].y);
    }
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
    if (!all(isfinite(float4(gh, gl, dh, dl))) || !isfinite(direction) || !isfinite(dot_high) || !isfinite(dot_low))
        atomic_store_explicit(status, 1u, memory_order_relaxed);
}
)METAL";
