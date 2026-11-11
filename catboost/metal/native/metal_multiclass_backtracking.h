#pragma once

// Append after CBMMetalMulticlassMathSource. The host owns CUDA's walker from
// methods/leaves_estimation/descent_helpers.cpp and step_estimator.cpp:
//   AnyImprovement: trial_value >= current_value (equality IS accepted).
//   Armijo: trial_value >= current_value + 1e-5 * step * direction_dot.
// Every trial consumes one leaf-estimation iteration; before the first accepted
// move CUDA allows up to 100 trials. Rejections halve the step; acceptance resets
// it to one and computes a new direction. One iteration bypasses backtracking.
// Values are NEGATIVE UNNORMALIZED weighted losses. OneVsAll divides its loss
// by C, but neither its gradients nor the Armijo dot product is divided by C.
// The multiclass pointwise CUDA oracle applies neither total-weight
// normalization nor ridge loss/gradient, even when lambda is in the Hessian.
static const char* CBMMetalMulticlassBacktrackingSource = R"METAL(

struct MulticlassBacktrackingParams {
    float step;
    uint type, reserved0, reserved1; // 1=AnyImprovement, 2=Armijo; host acceptance
};

// Requires exactly 256 threads; call uniformly after initializing scratch.
inline void MulticlassBacktrackingReduce256(threadgroup float2* scratch, uint tid) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 128; distance > 0; distance >>= 1) {
        if (tid < distance) {
            float high = scratch[tid].x, low = scratch[tid].y;
            MulticlassAdd(high, low, scratch[tid + distance].x);
            MulticlassAdd(high, low, scratch[tid + distance].y);
            scratch[tid] = float2(high, low);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Buffers: leaf statistics[leaves,S], directions[leaves,D], dots[leaves] as
// float4(high,low,base2_exponent,0), MulticlassParams. Dispatch leaves groups
// of 256 threads. Host reads ldexp(double(high)+double(low), int(exponent)).
// A valid float gradient times a valid float direction can exceed float range
// while CUDA's double Armijo dot remains finite. Factor both operands first,
// reduce scaled products, and restore the exponent only in host double math.
// MultiClass's full gradient sums to zero, so with gauge-relative direction
// d[k]=delta[k]-delta[C-1], the full dot product is sum(k<C-1, gradient[k]*d[k]).
// Reconstructing an extra last-class term here would count its effect twice.
// Do NOT mask small-weight leaves here: CUDA regularizes the candidate only
// after building the direction and its dot product. To mirror this, the host
// calls MulticlassSolveLeaves with a copy of params.min_leaf_weight=0, while
// the candidate builder below receives the original threshold (1e-20).
kernel void MulticlassBacktrackingDirectionDot(
    const device float* statistics [[buffer(0)]],
    const device float* directions [[buffer(1)]],
    device float4* dots [[buffer(2)]],
    constant MulticlassParams& p [[buffer(3)]],
    uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const uint dimensions = MulticlassDimension(p);
    const device float* gradient = statistics + leaf * MulticlassStatsWidth(p) + 1;
    const device float* direction = directions + leaf * dimensions;
    // Exponents from frexp are exact integers, including for subnormals. The
    // sentinels distinguish all-zero products from invalid input components.
    const int empty_exponent = -2147483647;
    const int invalid_exponent = 2147483647;
    int maximum_exponent = empty_exponent;
    for (uint k = tid; k < dimensions; k += 256) {
        if (!isfinite(gradient[k]) || !isfinite(direction[k])) {
            maximum_exponent = invalid_exponent;
            continue;
        }
        if (gradient[k] == 0.0f || direction[k] == 0.0f) continue;
        int left_exponent, right_exponent;
        frexp(gradient[k], left_exponent);
        frexp(direction[k], right_exponent);
        maximum_exponent = max(maximum_exponent, left_exponent + right_exponent);
    }
    threadgroup int exponents[256];
    exponents[tid] = maximum_exponent;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 128; distance > 0; distance >>= 1) {
        if (tid < distance) exponents[tid] = max(exponents[tid], exponents[tid + distance]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    maximum_exponent = exponents[0];
    if (maximum_exponent == invalid_exponent || maximum_exponent == empty_exponent) {
        if (tid == 0) dots[leaf] = maximum_exponent == empty_exponent
            ? float4(0.0f) : float4(MulticlassNaN(), MulticlassNaN(), 0.0f, 0.0f);
        return;
    }
    float high = 0.0f, low = 0.0f;
    for (uint k = tid; k < dimensions; k += 256) {
        if (gradient[k] == 0.0f || direction[k] == 0.0f) continue;
        int left_exponent, right_exponent;
        const float left = frexp(gradient[k], left_exponent);
        const float right = frexp(direction[k], right_exponent);
        const int exponent = left_exponent + right_exponent - maximum_exponent;
        const float product = left * right;
        MulticlassAdd(high, low, ldexp(product, exponent));
        MulticlassAdd(high, low, ldexp(fma(left, right, -product), exponent));
    }
    threadgroup float2 scratch[256];
    scratch[tid] = float2(high, low);
    MulticlassBacktrackingReduce256(scratch, tid);
    if (tid == 0) dots[leaf] = float4(scratch[0], float(maximum_exponent), 0.0f);
}

// Buffers: current_values[leaves,D], directions[leaves,D], statistics[leaves,S],
// trial_values[leaves,D], MulticlassParams, MulticlassBacktrackingParams.
// Dispatch leaves*D threads. Current and trial buffers MUST be distinct.
// CUDA adds double(step)*float(direction) to its float point and rounds once;
// an fma preserves that behavior for the walker's exact power-of-two steps.
kernel void MulticlassBacktrackingBuildCandidate(
    const device float* current_values [[buffer(0)]],
    const device float* directions [[buffer(1)]],
    const device float* statistics [[buffer(2)]],
    device float* trial_values [[buffer(3)]],
    constant MulticlassParams& p [[buffer(4)]],
    constant MulticlassBacktrackingParams& b [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    const uint dimensions = MulticlassDimension(p);
    if (index >= p.leaves * dimensions) return;
    const uint leaf = index / dimensions;
    const float weight = statistics[leaf * MulticlassStatsWidth(p)];
    trial_values[index] = weight < p.min_leaf_weight ? 0.0f
        : fma(b.step, directions[index], current_values[index]);
}

// Positive, unweighted CUDA row loss evaluated on Base + trial leaf point.
// A nonfinite trial value is preserved so the host rejects and halves the step
// while retaining the accepted point. No cursor or derivative buffers change.
inline float MulticlassBacktrackingRowLoss(
    uint row, uint target,
    const device float* base,
    const device float* point,
    constant MulticlassParams& p) {
    const uint dimensions = MulticlassDimension(p);
    if (target >= p.classes) return MulticlassNaN();
    if (p.objective == 0) {
        float maximum = 0.0f;
        for (uint k = 0; k < dimensions; ++k) {
            const float value = base[k * p.rows + row] + point[k];
            if (!isfinite(value)) return MulticlassNaN();
            maximum = max(maximum, value);
        }
        float denominator = exp(-maximum), tail = 0.0f;
        for (uint k = 0; k < dimensions; ++k) {
            const float value = base[k * p.rows + row] + point[k];
            MulticlassAdd(denominator, tail, exp(value - maximum));
        }
        const float target_value = target < dimensions
            ? base[target * p.rows + row] + point[target] : 0.0f;
        return (maximum - target_value) + log(denominator + tail);
    }
    float loss = 0.0f, tail = 0.0f;
    for (uint k = 0; k < dimensions; ++k) {
        const float value = base[k * p.rows + row] + point[k];
        if (!isfinite(value)) return MulticlassNaN();
        const float label = target == k ? 1.0f : 0.0f;
        MulticlassAdd(loss, tail, (max(value, 0.0f) - label * value)
            + MulticlassLogOnePlus(exp(-abs(value))));
    }
    return (loss + tail) / float(p.classes);
}

// Buffers: labels[N], original weights[N], Base[D,N], leaf_ids[N],
// trial_values[leaves,D], float2 value_partials[groups], MulticlassParams.
// Dispatch min(256,ceil(N/256)) (or another positive count) groups of 256.
// Host widens and sums both float components. The result is F=-sum(weight*loss)
// with NO total-weight normalization and NO lambda*point^2 penalty. Trial
// evaluation is read-only with respect to the accepted training state.
kernel void MulticlassBacktrackingReduceObjective(
    const device uint* labels [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device float* base [[buffer(2)]],
    const device uint* leaf_ids [[buffer(3)]],
    const device float* trial_values [[buffer(4)]],
    device float2* value_partials [[buffer(5)]],
    constant MulticlassParams& p [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]],
    uint groups [[threadgroups_per_grid]]) {
    const uint dimensions = MulticlassDimension(p);
    float high = 0.0f, low = 0.0f;
    for (uint row = group * 256 + tid; row < p.rows; row += groups * 256) {
        const float weight = weights[row];
        if (weight == 0.0f) continue;
        const float loss = MulticlassBacktrackingRowLoss(row, labels[row], base,
            trial_values + leaf_ids[row] * dimensions, p);
        MulticlassAdd(high, low, -weight * loss);
    }
    threadgroup float2 scratch[256];
    scratch[tid] = float2(high, low);
    MulticlassBacktrackingReduce256(scratch, tid);
    if (tid == 0) value_partials[group] = scratch[0];
}

// Buffers: Base[D,N], accepted_values[leaves,D], leaf_ids[N], cursor[D,N],
// MulticlassParams. Dispatch N*D threads AFTER acceptance (or use a separate
// trial cursor); rejected trials must leave the accepted cursor unchanged.
// Values here are unshrunk leaf points. Apply learning_rate only after the
// complete leaf search, using the training runtime's final cursor builder.
kernel void MulticlassBacktrackingBuildCursor(
    const device float* base [[buffer(0)]],
    const device float* accepted_values [[buffer(1)]],
    const device uint* leaf_ids [[buffer(2)]],
    device float* cursor [[buffer(3)]],
    constant MulticlassParams& p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    const uint dimensions = MulticlassDimension(p);
    if (index >= p.rows * dimensions) return;
    const uint row = index % p.rows, k = index / p.rows;
    cursor[index] = base[index] + accepted_values[leaf_ids[row] * dimensions + k];
}
)METAL";
