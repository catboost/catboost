#pragma once

// CUDA sources: targets/kernel/multilogit.cu and
// methods/leaves_estimation/{pointwise_oracle,descent_helpers}.cpp.
// The source can compile independently or follow the scalar Metal sources.
// Logits and row derivatives are CLASS-MAJOR. Leaf statistics and directions
// are LEAF-MAJOR. MultiClass stores C-1 logits: class C-1 has fixed logit zero.
static const char* CBMMetalMulticlassMathSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct MulticlassParams {
    uint rows, classes, objective, leaves; // objective: 0=MultiClass, 1=OneVsAll
    float l2, min_leaf_weight;
    uint leaf_method, reserved; // leaf_method: 0=Newton, 1=Gradient
};

inline uint MulticlassDimension(constant MulticlassParams& p) {
    return p.classes - (p.objective == 0 ? 1u : 0u);
}

inline uint MulticlassStatsWidth(constant MulticlassParams& p) {
    return 1 + p.classes + (p.objective == 0
        ? p.classes * (p.classes + 1) / 2 : p.classes);
}

inline float MulticlassNaN() {
    return as_type<float>(0x7fc00000u);
}

// Keep both components through all reduction levels. Shader compilation must
// disable fast math, because algebraic reassociation destroys the expansion.
inline void MulticlassAdd(thread float& high, thread float& low, float value) {
    const float sum = high + value;
    const float part = sum - high;
    const float error = (high - (sum - part)) + (value - part);
    const float tail = low + error;
    const float next = sum + tail;
    const float tail_part = next - sum;
    low = (sum - (next - tail_part)) + (tail - tail_part);
    high = next;
}

inline float MulticlassLogOnePlus(float value) {
    const float sum = 1.0f + value;
    return sum == 1.0f ? value : log(sum) - ((sum - 1.0f) - value) / sum;
}

// Buffers: labels[N], weights[N], logits[D*N], gradients[D*N],
// probabilities[C*N], positive weighted losses[N], params. Dispatch N threads.
// OVA's loss is averaged over classes; its derivatives and Hessian are not.
kernel void MulticlassDerivatives(
    const device uint* labels [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device float* logits [[buffer(2)]],
    device float* gradients [[buffer(3)]],
    device float* probabilities [[buffer(4)]],
    device float* losses [[buffer(5)]],
    constant MulticlassParams& p [[buffer(6)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint dimensions = MulticlassDimension(p);
    const float weight = weights[row];
    // A masked row must make no contribution even if an intermediate cursor
    // overflows. Host input validation still rejects nonfinite input cursors.
    if (weight == 0.0f) {
        for (uint k = 0; k < dimensions; ++k) gradients[k * p.rows + row] = 0.0f;
        for (uint k = 0; k < p.classes; ++k) probabilities[k * p.rows + row] = 0.0f;
        losses[row] = 0.0f;
        return;
    }
    bool valid = isfinite(weight) && weight >= 0.0f && labels[row] < p.classes;
    float maximum = 0.0f;
    for (uint k = 0; k < dimensions; ++k) {
        const float value = logits[k * p.rows + row];
        valid = valid && isfinite(value);
        maximum = max(maximum, value);
    }
    if (!valid) {
        for (uint k = 0; k < dimensions; ++k) gradients[k * p.rows + row] = MulticlassNaN();
        for (uint k = 0; k < p.classes; ++k) probabilities[k * p.rows + row] = MulticlassNaN();
        losses[row] = MulticlassNaN();
        return;
    }
    if (p.objective == 0) {
        float sum = exp(-maximum), tail = 0.0f;
        for (uint k = 0; k < dimensions; ++k) {
            MulticlassAdd(sum, tail, exp(logits[k * p.rows + row] - maximum));
        }
        const float denominator = sum + tail;
        for (uint k = 0; k < dimensions; ++k) {
            const float probability = exp(logits[k * p.rows + row] - maximum) / denominator;
            probabilities[k * p.rows + row] = probability;
            gradients[k * p.rows + row] = weight * ((labels[row] == k ? 1.0f : 0.0f) - probability);
        }
        probabilities[dimensions * p.rows + row] = exp(-maximum) / denominator;
        const float target_logit = labels[row] < dimensions
            ? logits[labels[row] * p.rows + row] : 0.0f;
        losses[row] = weight * ((maximum - target_logit) + log(denominator));
    } else {
        float loss = 0.0f, tail = 0.0f;
        for (uint k = 0; k < dimensions; ++k) {
            const float value = logits[k * p.rows + row];
            const float exponential = exp(-abs(value));
            const float probability = clamp(value >= 0.0f
                ? 1.0f / (1.0f + exponential) : exponential / (1.0f + exponential),
                1.0e-7f, 1.0f - 1.0e-7f); // CUDA ClipProb
            const float target = labels[row] == k ? 1.0f : 0.0f;
            probabilities[k * p.rows + row] = probability;
            gradients[k * p.rows + row] = weight * (target - probability);
            MulticlassAdd(loss, tail, (max(value, 0.0f) - target * value)
                + MulticlassLogOnePlus(exponential));
        }
        losses[row] = weight * ((loss + tail) / float(p.classes));
    }
}

// One group per (leaf, statistic), exactly 256 threads per group. Host provides
// the partition's row permutation and offsets; no per-row C*C Hessian buffer
// is needed. S = 1+C+C*(C+1)/2 for MultiClass or 1+2*C for OneVsAll.
// Each leaf holds: [weight, C gradients, packed-lower Hessian/OVA diagonal].
// MultiClass's last gradient is subsequently reconstructed from the reduced
// first C-1 entries by the solver, matching the CUDA leaf oracle exactly.
kernel void MulticlassReduceLeafStats(
    const device float* gradients [[buffer(0)]],
    const device float* probabilities [[buffer(1)]],
    const device float* weights [[buffer(2)]],
    const device uint* row_indices [[buffer(3)]],
    const device uint* offsets [[buffer(4)]],
    device float* statistics [[buffer(5)]],
    constant MulticlassParams& p [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    const uint width = MulticlassStatsWidth(p);
    const uint leaf = group / width, component = group % width;
    if (leaf >= p.leaves) return;
    uint matrix_row = 0, matrix_column = 0;
    if (component > p.classes) {
        const uint packed = component - 1 - p.classes;
        if (p.objective == 0) {
            while ((matrix_row + 1) * (matrix_row + 2) / 2 <= packed) ++matrix_row;
            matrix_column = packed - matrix_row * (matrix_row + 1) / 2;
        } else {
            matrix_row = matrix_column = packed;
        }
    }
    float high = 0.0f, low = 0.0f;
    for (uint position = offsets[leaf] + tid; position < offsets[leaf + 1]; position += 256) {
        const uint row = row_indices[position];
        const float weight = weights[row];
        float value = 0.0f;
        if (component == 0) {
            value = weight;
        } else if (component <= p.classes) {
            const uint k = component - 1;
            if (p.objective == 0 && k == p.classes - 1) {
                float sum = 0.0f, tail = 0.0f;
                for (uint d = 0; d < k; ++d) MulticlassAdd(sum, tail, gradients[d * p.rows + row]);
                value = -(sum + tail);
            } else {
                value = gradients[k * p.rows + row];
            }
        } else if (weight != 0.0f) {
            const float probability = probabilities[matrix_row * p.rows + row];
            value = matrix_column == matrix_row
                ? weight * (1.0f - probability) * probability
                : -weight * probabilities[matrix_column * p.rows + row] * probability;
        }
        MulticlassAdd(high, low, value);
    }
    threadgroup float2 scratch[256];
    scratch[tid] = float2(high, low);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 128; distance > 0; distance >>= 1) {
        if (tid < distance) {
            float sum = scratch[tid].x, tail = scratch[tid].y;
            MulticlassAdd(sum, tail, scratch[tid + distance].x);
            MulticlassAdd(sum, tail, scratch[tid + distance].y);
            scratch[tid] = float2(sum, tail);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) statistics[group] = scratch[0].x + scratch[0].y;
}

// Dispatch one thread per leaf. Workspace is leaves*(C*C+C) floats; the matrix
// is device-backed, avoiding a fixed-size private array and class-count limit.
// Output: unscaled incremental direction [leaves,D], no accumulated leaf point.
// Status per leaf: 0=success, 1=nonfinite statistics, 2=nonpositive Cholesky
// pivot, 3=nonfinite direction. Empty/underweight leaves emit zeros successfully.
// MultiClass Newton is CUDA's full C-dimensional (H+lambda I) solve, expressed
// in the last-class gauge without its null mode: let d[i]=step[i]-step[C-1].
// Since sum(step)=0, step[i]=d[i]-sum(d)/C and step[C-1]=-sum(d)/C. Thus solve
// (H_top + lambda*(I - 11^T/C))*d = gradient_top. The dense negative lambda/C
// term is essential: simply adding lambda I to H_top changes regularization.
// This equivalent coupled system also avoids losing a small positive lambda
// to the full Hessian's float32 gauge cancellation. Lambda zero is well-defined
// whenever the leading principal Hessian is positive definite.
kernel void MulticlassSolveLeaves(
    const device float* statistics [[buffer(0)]],
    device float* workspace [[buffer(1)]],
    device float* directions [[buffer(2)]],
    device uint* status [[buffer(3)]],
    constant MulticlassParams& p [[buffer(4)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const uint c = p.classes, dimensions = MulticlassDimension(p);
    const device float* stats = statistics + leaf * MulticlassStatsWidth(p);
    device float* matrix = workspace + leaf * (c * c + c);
    device float* solution = matrix + c * c;
    device float* result = directions + leaf * dimensions;
    status[leaf] = 0;
    for (uint i = 0; i < dimensions; ++i) result[i] = 0.0f;
    const float weight = stats[0];
    if (!isfinite(weight) || weight < 0.0f) { status[leaf] = 1; return; }
    if (weight == 0.0f || weight < p.min_leaf_weight) return;
    float gradient_sum = 0.0f, gradient_tail = 0.0f;
    bool has_gradient = false;
    for (uint i = 0; i < dimensions; ++i) {
        solution[i] = stats[1 + i];
        if (!isfinite(solution[i])) { status[leaf] = 1; return; }
        has_gradient = has_gradient || solution[i] != 0.0f;
        MulticlassAdd(gradient_sum, gradient_tail, solution[i]);
    }
    // A saturated, already-correct leaf can have zero Hessian and gradient.
    // Its zero direction is defined without factorizing a singular matrix.
    if (!has_gradient) return;
    if (p.objective == 0) solution[c - 1] = -(gradient_sum + gradient_tail);
    if (p.leaf_method == 1 || p.objective == 1) {
        for (uint i = 0; i < c; ++i) {
            const float curvature = p.leaf_method == 1 ? weight : stats[1 + c + i];
            const float denominator = curvature + p.l2;
            if (!isfinite(denominator) || curvature < 0.0f) { status[leaf] = 1; return; }
            solution[i] = denominator > 0.0f ? solution[i] / (denominator + 1.0e-20f) : 0.0f;
        }
    } else {
        const uint n = c - 1;
        solution[c - 1] = 0.0f; // the solve directly returns baseline differences
        for (uint i = 0; i < n; ++i) {
            for (uint j = 0; j <= i; ++j) {
                const float value = stats[1 + c + i * (i + 1) / 2 + j]
                    + p.l2 * ((i == j ? 1.0f : 0.0f) - 1.0f / float(c));
                if (!isfinite(value)) { status[leaf] = 1; return; }
                matrix[i * c + j] = value;
            }
        }
        // In-place lower Cholesky with compensated inner products.
        for (uint i = 0; i < n; ++i) {
            for (uint j = 0; j <= i; ++j) {
                float sum = matrix[i * c + j], tail = 0.0f;
                for (uint k = 0; k < j; ++k) MulticlassAdd(sum, tail, -matrix[i * c + k] * matrix[j * c + k]);
                const float residual = sum + tail;
                if (!isfinite(residual)) { status[leaf] = 1; return; }
                if (i == j) {
                    if (!(residual > 0.0f)) { status[leaf] = 2; return; }
                    matrix[i * c + j] = sqrt(residual);
                } else {
                    matrix[i * c + j] = residual / matrix[j * c + j];
                }
            }
        }
        for (uint i = 0; i < n; ++i) {
            float sum = solution[i], tail = 0.0f;
            for (uint k = 0; k < i; ++k) MulticlassAdd(sum, tail, -matrix[i * c + k] * solution[k]);
            solution[i] = (sum + tail) / matrix[i * c + i];
        }
        for (uint reverse = 0; reverse < n; ++reverse) {
            const uint i = n - 1 - reverse;
            float sum = solution[i], tail = 0.0f;
            for (uint k = i + 1; k < n; ++k) MulticlassAdd(sum, tail, -matrix[k * c + i] * solution[k]);
            solution[i] = (sum + tail) / matrix[i * c + i];
        }
    }
    const float baseline = p.objective == 0 ? solution[c - 1] : 0.0f;
    for (uint i = 0; i < dimensions; ++i) {
        const float direction = solution[i] - baseline;
        if (!isfinite(direction)) { status[leaf] = 3; return; }
    }
    for (uint i = 0; i < dimensions; ++i) result[i] = solution[i] - baseline;
}
)METAL";
