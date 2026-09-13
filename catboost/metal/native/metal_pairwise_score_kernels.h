#pragma once

// Append after CBMMetalLeafMatrixSource: this uses its compensated arithmetic
// and LeafMatrixParams. CUDA split stabilization differs from leaf estimation.
static const char* CBMMetalPairwiseScoreSource = R"METAL(

// linear_solver.cu::RegularizeImpl. The average excludes zero/near-zero
// diagonals from its divisor, while the numerator retains the full trace.
// Input and output use a complete row-major physical stride of leaves.
kernel void RegularizePairwiseSplitMatrix(const device float* hessian [[buffer(0)]],
    device float2* workspace [[buffer(1)]], constant LeafMatrixParams& p [[buffer(2)]],
    uint tid [[thread_position_in_threadgroup]], uint matrix_id [[threadgroup_position_in_grid]]) {
    hessian += matrix_id * p.leaves * p.leaves;
    workspace += matrix_id * p.leaves * p.leaves;
    threadgroup float average;
    if (tid == 0) {
        float trace = 0.0f, count = 0.0f;
        for (uint row = 0; row < p.leaves; ++row) {
            const float diagonal = hessian[row * p.leaves + row];
            trace += diagonal;
            count += float(diagonal > 1e-9f);
        }
        average = count > 0.0f ? trace / count : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = LeafMatrixDimension(p);
    const float2 prior = LeafMatrixDivide(float2(p.non_diag_l2, 0.0f), float2(float(p.leaves), 0.0f));
    for (uint cell = tid; cell < p.leaves * p.leaves; cell += 256) {
        const uint row = cell / p.leaves, column = cell % p.leaves;
        if (row >= n || column >= n) { workspace[cell] = float2(0.0f); continue; }
        float2 value = float2(hessian[cell], 0.0f);
        if (row == column) {
            if (hessian[cell] <= 1e-7f) value = LeafMatrixAdd(value, float2(average + 0.1f, 0.0f));
            value = LeafMatrixAdd(value, float2(0.05f * average + 1e-20f, 0.0f));
            value = LeafMatrixAdd(value, float2(p.l2, 0.0f));
            value = LeafMatrixAdd(value, LeafMatrixAdd(float2(p.non_diag_l2, 0.0f), -prior));
        } else value = LeafMatrixAdd(value, -prior);
        workspace[cell] = value;
    }
}

// CUDA calls ZeroMean only when removing the final coordinate. The full QCE
// system has a point diagonal and its solution must retain its absolute mean.
kernel void CenterPairwiseSplitSolution(device float* direction [[buffer(0)]],
    constant LeafMatrixParams& p [[buffer(1)]], uint tid [[thread_position_in_threadgroup]],
    uint matrix_id [[threadgroup_position_in_grid]]) {
    direction += matrix_id * p.leaves;
    if (p.has_diagonal_part) return;
    threadgroup float2 sums[256];
    const float value = tid + 1 < p.leaves ? direction[tid] : 0.0f;
    sums[tid] = float2(value / float(p.leaves), 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) sums[tid] = LeafMatrixAdd(sums[tid], sums[tid + width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < p.leaves) {
        const float2 centered = LeafMatrixAdd(float2(value, 0.0f), -sums[0]);
        direction[tid] = centered.x + centered.y;
    }
}

// CalcScoresCholeskyImpl evaluates ORIGINAL H, before split stabilization or
// ridge. The positive improvement score is beta.G - .5*beta.H.beta. It is not
// interchangeable with .5*beta.G because beta solves a regularized system.
kernel void ScorePairwiseSplitSolution(const device float* hessian [[buffer(0)]],
    const device float* gradient [[buffer(1)]], const device float* solution [[buffer(2)]],
    device float2* score [[buffer(3)]], constant LeafMatrixParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint matrix_id [[threadgroup_position_in_grid]]) {
    hessian += matrix_id * p.leaves * p.leaves;
    gradient += matrix_id * p.leaves;
    solution += matrix_id * p.leaves;
    score += matrix_id;
    threadgroup float2 terms[256];
    float2 term = float2(0.0f);
    if (tid < p.leaves) {
        float2 product = float2(0.0f);
        for (uint column = 0; column < p.leaves; ++column)
            product = LeafMatrixAdd(product, LeafMatrixMultiply(float2(hessian[tid * p.leaves + column], 0.0f),
                float2(solution[column], 0.0f)));
        const float2 residual = LeafMatrixAdd(float2(gradient[tid], 0.0f),
            -LeafMatrixMultiply(float2(0.5f, 0.0f), product));
        term = LeafMatrixMultiply(float2(solution[tid], 0.0f), residual);
    }
    terms[tid] = term;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) terms[tid] = LeafMatrixAdd(terms[tid], terms[tid + width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) score[0] = terms[0];
}

struct PairwiseSelectionParams {
    uint candidates, features, reserved0, reserved1;
    float previous_negative_score, reserved2, reserved3, reserved4;
};

// Simple leaves are the selected weak-target split solution, with the original
// matrix diagonal as model weights. Convert adjacent solver children to the
// model's highest leaf bit; CUDA reverses the final one-hot predicate.
kernel void ExportSimplePairwiseLeaves(const device float* direction [[buffer(0)]],
    const device float* hessian [[buffer(1)]], device float* values [[buffer(2)]],
    device float* weights [[buffer(3)]], constant LeafMatrixParams& p [[buffer(4)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const uint parents = p.leaves / 2;
    const uint source = 2 * (leaf % parents) + ((leaf / parents) ^ p.reserved0);
    values[leaf] = direction[source];
    weights[leaf] = hessian[source * p.leaves + source];
}

// One 256-thread group. Return the first maximum-gain candidate and the
// NEGATIVE raw score, matching split_pairwise.cu's persistent sign convention.
// Winner=UINT_MAX if all candidates are nonfinite/invalid; caller validates it.
kernel void SelectPairwiseSplitWinner(const device float2* scores [[buffer(0)]],
    const device uint* features [[buffer(1)]], const device float* feature_weights [[buffer(2)]],
    device uint* winner [[buffer(3)]], device float2* selected_score_gain [[buffer(4)]],
    constant PairwiseSelectionParams& p [[buffer(5)]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float gains[256], raw_scores[256];
    threadgroup uint indices[256];
    float best_gain = -INFINITY, best_score = -INFINITY;
    uint best = 0xffffffffu;
    for (uint index = tid; index < p.candidates; index += 256) {
        if (features[index] >= p.features) continue;
        const float score = scores[index].x + scores[index].y;
        const uint feature = features[index];
        const float weight = p.reserved0 ? feature_weights[2 * feature] * feature_weights[2 * feature + 1]
            : feature_weights[feature];
        const float2 shifted = LeafMatrixAdd(float2(score, 0.0f), float2(p.previous_negative_score, 0.0f));
        const float2 product = LeafMatrixMultiply(shifted, float2(weight, 0.0f));
        const float gain = product.x + product.y;
        if (!isfinite(score) || !isfinite(gain) || !isfinite(weight) || weight < 0.0f) continue;
        if (gain > best_gain || (gain == best_gain && index < best)) {
            best_gain = gain; best_score = score; best = index;
        }
    }
    gains[tid] = best_gain; raw_scores[tid] = best_score; indices[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width && (gains[tid + width] > gains[tid]
                || (gains[tid + width] == gains[tid] && indices[tid + width] < indices[tid]))) {
            gains[tid] = gains[tid + width]; raw_scores[tid] = raw_scores[tid + width]; indices[tid] = indices[tid + width];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { winner[0] = indices[0]; selected_score_gain[0] = float2(-raw_scores[0], gains[0]); }
}
)METAL";
