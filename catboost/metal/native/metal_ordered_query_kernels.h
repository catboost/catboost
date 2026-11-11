#pragma once

// Append after the shared objective/backtracking, Ordered session and
// querywise sources. Query derivatives and original sample weights use the
// flattened task cursor layout; leaf ids retain their original-row banks.
// Every reduction below requires one complete 256-thread group per output.
static const char* CBMMetalOrderedQuerySource = R"METAL(

inline void OrderedQueryReduceExpansion(threadgroup float2* scratch, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            float high = scratch[tid].x, low = scratch[tid].y;
            ObjectiveAddExpansion(high, low, scratch[tid + stride].x);
            ObjectiveAddExpansion(high, low, scratch[tid + stride].y);
            scratch[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Dispatch rows x tasks. Mode 0 prepares structure derivatives, mode 1 the
// complete current/trial leaf point, and mode 2 the published model point.
// A trial can overflow and must remain rejectable without poisoning status.
kernel void OrderedQueryPreparePoint(const device float* cursors [[buffer(0)]],
    const device float* published [[buffer(1)]], const device float* values [[buffer(2)]],
    const device uint* permutations [[buffer(3)]], const device uint* leaf_ids [[buffer(4)]],
    const device uint4* tasks [[buffer(5)]], device float* point [[buffer(6)]],
    device atomic_uint* status [[buffer(7)]], constant OrderedTrainingParams& p [[buffer(8)]],
    constant OrderedStepParams& step [[buffer(9)]], constant uint& mode [[buffer(10)]],
    uint2 position [[thread_position_in_grid]]) {
    if (position.y >= step.tasks) return;
    const uint4 task = tasks[position.y];
    if (position.x >= task.y) return;
    const uint row = permutations[task.w * p.rows + position.x], slot = task.z + position.x;
    float value = mode == 2 ? published[row] : cursors[slot];
    if (mode == 1) {
        const uint leaf = leaf_ids[task.w * p.reserved0 + row];
        if (leaf < step.leaves) value += values[position.y * (1u << p.depth) + leaf];
        else {
            value = ObjectiveInvalidValue();
            atomic_store_explicit(status, 1u, memory_order_relaxed);
        }
    }
    point[slot] = value;
    if (mode > 2 || (mode != 1 && !isfinite(value)))
        atomic_store_explicit(status, 1u, memory_order_relaxed);
}

// Dispatch cursor_count rows. Curvature is relevant only for Newton scores:
// Gradient scoring must tolerate a finite query objective with negative raw
// curvature, just as the scalar query implementation does.
kernel void OrderedQueryPublishDerivatives(const device float* gradients [[buffer(0)]],
    const device float* hessian [[buffer(1)]], const device float* weights [[buffer(2)]],
    device float2* derivatives [[buffer(3)]], device atomic_uint* status [[buffer(4)]],
    constant OrderedTrainingParams& p [[buffer(5)]], constant OrderedStepParams& step [[buffer(6)]],
    uint slot [[thread_position_in_grid]]) {
    if (slot >= step.cursor_count) return;
    const float2 value = float2(gradients[slot], p.score_function ? hessian[slot] : weights[slot]);
    derivatives[slot] = value;
    // Combination and Simple QuerySoftMax can have finite signed weak
    // weights. Simple still estimates Gradient1 leaves from original weights.
    const bool signed_weights = p.objective == 19 || (p.objective == 13 && p.reserved2);
    if (!all(isfinite(value)) || (!signed_weights && value.y < 0.0f))
        atomic_store_explicit(status, 1u, memory_order_relaxed);
}

// Dispatch leaves x tasks full groups. The query oracle has already computed
// every derivative jointly at this task's complete point. Project only its
// estimation prefix, retaining original sample mass for Gradient leaves,
// empty-leaf regularization and the optional task normalization.
kernel void OrderedQueryEstimateLeaves(const device float* gradients [[buffer(0)]],
    const device float* hessian [[buffer(1)]], const device float* weights [[buffer(2)]],
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
        const uint slot = task.z + position, row = permutations[task.w * p.rows + position];
        const float weight = weights[slot];
        ObjectiveAddExpansion(mass_high, mass_low, weight);
        if (leaf_ids[task.w * p.reserved0 + row] != leaf) continue;
        ObjectiveAddExpansion(high, low,
            float3(gradients[slot], p.leaf_method ? weight : hessian[slot], weight));
    }
    high_scratch[tid] = float4(high, 0); low_scratch[tid] = float4(low, 0);
    mass_scratch[tid] = float2(mass_high, mass_low);
    ObjectiveReduceExpansions(high_scratch, low_scratch, tid);
    OrderedQueryReduceExpansion(mass_scratch, tid);
    if (!tid) {
        const float3 s = high_scratch[0].xyz + low_scratch[0].xyz;
        const float mass = mass_scratch[0].x + mass_scratch[0].y;
        const float diagonal = s.y + p.l2 * (p.normalize ? mass : 1.0f);
        float next = point;
        if (s.z < 1e-20f) next = 0.0f;
        else if (diagonal > 0.0f) {
            const float gradient = p.reserved1
                ? s.x - p.l2 * (p.normalize ? mass : 1.0f) * point : s.x;
            next += gradient / (diagonal + 1e-20f);
        }
        values[output] = next; leaf_weights[output] = s.z;
        // QuerySoftMax permits negative raw curvature when regularization
        // still yields a positive Newton diagonal on each nonempty leaf.
        // Combination also permits nonpositive finite diagonals: CUDA keeps
        // the existing point and gives that coordinate a zero direction.
        if (!all(isfinite(s)) || !isfinite(mass) || mass < 0.0f || !isfinite(diagonal) ||
            !isfinite(point) || !isfinite(next) || s.z < 0.0f ||
            (p.objective != 19 && !p.leaf_method && s.z >= 1e-20f && diagonal <= 0.0f))
            atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

// The walker retains one direction and dot-product expansion per task leaf.
// Do not zero directions on weight-empty leaves: candidate regularization
// follows direction construction, as in OrderedBacktrackingDirections.
kernel void OrderedQueryBacktrackingDirections(const device float* gradients [[buffer(0)]],
    const device float* hessian [[buffer(1)]], const device float* weights [[buffer(2)]],
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
    threadgroup float4 high_scratch[256], low_scratch[256];
    threadgroup float2 mass_scratch[256];
    float3 high = 0.0f, low = 0.0f;
    float mass_high = 0.0f, mass_low = 0.0f;
    for (uint position = tid; position < task.x; position += 256) {
        const uint slot = task.z + position, row = permutations[task.w * p.rows + position];
        const float weight = weights[slot];
        ObjectiveAddExpansion(mass_high, mass_low, weight);
        if (leaf_ids[task.w * p.reserved0 + row] != leaf) continue;
        ObjectiveAddExpansion(high, low,
            float3(gradients[slot], p.leaf_method ? weight : hessian[slot], weight));
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
            const float2 g = mass > 0.0f ? BacktrackingDivideExpansion(gh, gl, mass) : float2(0);
            const float2 h = mass > 0.0f ? BacktrackingDivideExpansion(dh, dl, mass) : float2(0);
            gh = g.x; gl = g.y; dh = h.x; dl = h.y;
        }
        // reserved1 is populated only by the additive runtime ridge setter.
        // CUDA adds the penalty after optional task normalization.
        if (p.reserved1) {
            const float ridge = -p.l2 * values[output];
            ObjectiveAddExpansion(gh, gl, ridge);
            ObjectiveAddExpansion(gh, gl, fma(-p.l2, values[output], -ridge));
        }
        ObjectiveAddExpansion(dh, dl, p.l2);
        const float diagonal = dh + dl;
        float direction = 0.0f;
        if (diagonal > 0.0f) {
            ObjectiveAddExpansion(dh, dl, 1e-20f);
            direction = BacktrackingDirection(gh, gl, dh, dl);
        }
        directions[output] = direction;
        float dot_high = gh * direction, dot_low = fma(gh, direction, -dot_high);
        ObjectiveAddExpansion(dot_high, dot_low, gl * direction);
        direction_dot[output] = float2(dot_high, dot_low);
        if (!all(isfinite(sh + sl)) || !isfinite(mass) || mass < 0.0f || !isfinite(diagonal) ||
            !isfinite(values[output]) || !isfinite(gh) || !isfinite(gl) || !isfinite(direction) ||
            !isfinite(dot_high) || !isfinite(dot_low) || sh.z + sl.z < 0.0f ||
            (p.objective != 19 && !p.leaf_method && sh.z + sl.z >= 1e-20f && diagonal <= 0.0f))
            atomic_store_explicit(status, 1u, memory_order_relaxed);
    }
}

// Dispatch one group per task. Ranges contain (begin query, estimation end,
// quality end, 0); estimation prefixes end on complete-query boundaries.
// Stats carry positive query loss. Preserve nonfinite trial losses so the
// host walker can reject and halve its common step without setting status.
kernel void OrderedQueryBacktrackingObjective(const device float2* query_stats [[buffer(0)]],
    const device uint4* task_query_ranges [[buffer(1)]], const device float* task_mass [[buffer(2)]],
    device float2* loss [[buffer(3)]], constant OrderedTrainingParams& p [[buffer(4)]],
    constant OrderedStepParams& step [[buffer(5)]], uint tid [[thread_position_in_threadgroup]],
    uint task_id [[threadgroup_position_in_grid]]) {
    if (task_id >= step.tasks) return;
    const uint4 range = task_query_ranges[task_id];
    threadgroup float2 scratch[256];
    float high = 0.0f, low = 0.0f;
    for (uint query = range.x + tid; query < range.y; query += 256)
        ObjectiveAddExpansion(high, low, -query_stats[query].x);
    if (p.normalize) {
        const float2 normalized = task_mass[task_id] > 0.0f
            ? BacktrackingDivideExpansion(high, low, task_mass[task_id]) : float2(0);
        high = normalized.x; low = normalized.y;
    }
    scratch[tid] = float2(high, low);
    OrderedQueryReduceExpansion(scratch, tid);
    if (!tid) loss[task_id] = scratch[0];
}

// Dispatch one group per task after final leaf estimation, before shrinkage.
// Center every actual leaf coordinate, including empty leaves. Dividing
// before summation keeps the mean of finite same-sign extremes finite.
kernel void OrderedQueryCenterLeaves(device float* values [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]], constant OrderedTrainingParams& p [[buffer(2)]],
    constant OrderedStepParams& step [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint task_id [[threadgroup_position_in_grid]]) {
    if (task_id >= step.tasks) return;
    const uint begin = task_id * (1u << p.depth);
    threadgroup float2 scratch[256];
    float high = 0.0f, low = 0.0f;
    for (uint leaf = tid; leaf < step.leaves; leaf += 256) {
        const float value = values[begin + leaf];
        if (!isfinite(value)) atomic_store_explicit(status, 1u, memory_order_relaxed);
        ObjectiveAddExpansion(high, low, value / float(step.leaves));
    }
    scratch[tid] = float2(high, low);
    OrderedQueryReduceExpansion(scratch, tid);
    const float mean = scratch[0].x + scratch[0].y;
    for (uint leaf = tid; leaf < step.leaves; leaf += 256) {
        const float value = values[begin + leaf] - mean;
        if (!isfinite(value)) atomic_store_explicit(status, 1u, memory_order_relaxed);
        values[begin + leaf] = value;
    }
}
// Edge ranges are grouped by each whole-query occurrence in cursor order.
kernel void OrderedPairQueryStatistics(const device float4* edges [[buffer(0)]],
    const device uint2* ranges [[buffer(1)]], device float2* statistics [[buffer(2)]],
    constant uint& queries [[buffer(3)]], uint tid [[thread_position_in_threadgroup]],
    uint query [[threadgroup_position_in_grid]]) {
    if (query >= queries) return;
    threadgroup float4 high_scratch[256], low_scratch[256];
    float4 high = 0, low = 0;
    for (uint edge = ranges[query].x + tid; edge < ranges[query].y; edge += 256)
        QuerywiseAccumulate(float4(edges[edge].zw, 0, 0), high, low);
    high_scratch[tid] = high; low_scratch[tid] = low;
    QuerywiseReduceExpansions(high_scratch, low_scratch, tid);
    if (!tid) statistics[query] = high_scratch[0].xy + low_scratch[0].xy;
}
)METAL";
