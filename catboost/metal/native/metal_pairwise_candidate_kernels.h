#pragma once

// Append after metal_pairwise_matrix_kernels.h. A bounded candidate tile
// duplicates only (key,edge-id), then one stable radix sort groups all directed
// child-leaf cells. Work is O(candidate_count * edges + candidate_count * L^2),
// avoiding a scan of every edge for every Hessian cell.
static const char* CBMMetalPairwiseCandidateSource = R"METAL(
struct PairwiseCandidateParams {
    uint rows, pairs, parent_leaves, candidates;
    uint features, first_candidate, leaf_method, reserved;
};

// Candidate child order is CUDA solver order: 2*parent + predicate. One-hot
// predicates are inverted here; FixSolutionLeavesValuesLayout swaps that last
// bit when a Simple estimator exports the winning solution to model order.
kernel void BuildPairwiseCandidateKeys(const device uchar* bins [[buffer(0)]],
    const device uint* leaf_ids [[buffer(1)]], const device uint* winners [[buffer(2)]],
    const device uint* losers [[buffer(3)]], const device uint* features [[buffer(4)]],
    const device uint* borders [[buffer(5)]], const device uchar* types [[buffer(6)]],
    device uint* keys [[buffer(7)]], device uint* indices [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]], constant PairwiseCandidateParams& p [[buffer(10)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= p.candidates * p.pairs) return;
    const uint candidate = item / p.pairs, edge = item % p.pairs;
    const uint feature = features[p.first_candidate + candidate];
    const uint border = borders[p.first_candidate + candidate];
    const uint type = types[p.first_candidate + candidate];
    keys[item] = 0xffffffffu; indices[item] = edge;
    const uint a = winners[edge], b = losers[edge], leaves = 2 * p.parent_leaves;
    if (a >= p.rows || b >= p.rows || a == b || feature >= p.features || border > 255 || type > 1) {
        atomic_fetch_or_explicit(status + candidate, 1u, memory_order_relaxed); return;
    }
    if (leaf_ids[a] >= p.parent_leaves || leaf_ids[b] >= p.parent_leaves) {
        atomic_fetch_or_explicit(status + candidate, 1u, memory_order_relaxed); return;
    }
    const uint av = bins[feature * p.rows + a], bv = bins[feature * p.rows + b];
    const uint left = 2 * leaf_ids[a] + uint(type ? av != border : av > border);
    const uint right = 2 * leaf_ids[b] + uint(type ? bv != border : bv > border);
    if (left != right) keys[item] = candidate * leaves * leaves + left * leaves + right;
}

kernel void BuildPairwiseCandidateOffsets(const device uint* keys [[buffer(0)]],
    device uint* offsets [[buffer(1)]], constant PairwiseCandidateParams& p [[buffer(2)]],
    uint cell [[thread_position_in_grid]]) {
    const uint cells = p.candidates * 4 * p.parent_leaves * p.parent_leaves;
    if (cell > cells) return;
    uint first = 0, last = p.candidates * p.pairs;
    while (first < last) {
        const uint middle = first + (last - first) / 2;
        if (keys[middle] < cell) first = middle + 1;
        else last = middle;
    }
    offsets[cell] = first;
}

kernel void ReducePairwiseCandidateCells(const device uint* offsets [[buffer(0)]],
    const device uint* sorted_edges [[buffer(1)]], const device float4* edges [[buffer(2)]],
    device float4* cells [[buffer(3)]], constant PairwiseCandidateParams& p [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]], uint cell [[threadgroup_position_in_grid]]) {
    if (cell >= p.candidates * 4 * p.parent_leaves * p.parent_leaves) return;
    const uint begin = offsets[cell], end = offsets[cell + 1];
    if (begin == end) {
        if (!tid) { cells[2 * cell] = float4(0.0f); cells[2 * cell + 1] = float4(0.0f); }
        return;
    }
    threadgroup float4 high_parts[256], low_parts[256];
    float4 high = float4(0.0f), low = float4(0.0f);
    for (uint index = begin + tid; index < end; index += 256)
        PairMatrixAdd4(high, low, edges[sorted_edges[index]]);
    high_parts[tid] = high; low_parts[tid] = low;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 128; width; width >>= 1) {
        if (tid < width) {
            high = high_parts[tid]; low = low_parts[tid];
            PairMatrixAdd4(high, low, high_parts[tid + width]);
            PairMatrixAdd4(high, low, low_parts[tid + width]);
            high_parts[tid] = high; low_parts[tid] = low;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) { cells[2 * cell] = high_parts[0]; cells[2 * cell + 1] = low_parts[0]; }
}

kernel void AssemblePairwiseCandidateMatrices(const device float4* cells [[buffer(0)]],
    device float* gradients [[buffer(1)]], device float* hessians [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]], constant PairwiseCandidateParams& p [[buffer(4)]],
    uint cell [[thread_position_in_grid]]) {
    const uint leaves = 2 * p.parent_leaves, stride = leaves * leaves;
    if (cell >= p.candidates * stride) return;
    const uint candidate = cell / stride, local = cell % stride;
    const uint row = local / leaves, column = local % leaves;
    const uint mass = p.leaf_method == 1 ? 2 : 1;
    cells += 2 * candidate * stride;
    float4 high = float4(0.0f), low = float4(0.0f);
    if (row != column) {
        const uint reverse = column * leaves + row;
        PairMatrixAdd4(high, low, cells[2 * local]); PairMatrixAdd4(high, low, cells[2 * local + 1]);
        PairMatrixAdd4(high, low, cells[2 * reverse]); PairMatrixAdd4(high, low, cells[2 * reverse + 1]);
        hessians[cell] = -(high[mass] + low[mass]);
    } else {
        for (uint other = 0; other < leaves; ++other) {
            if (other == row) continue;
            const uint outgoing = row * leaves + other, incoming = other * leaves + row;
            PairMatrixAdd4(high, low, cells[2 * outgoing]); PairMatrixAdd4(high, low, cells[2 * outgoing + 1]);
            float4 ih = cells[2 * incoming], il = cells[2 * incoming + 1];
            ih.x = -ih.x; il.x = -il.x;
            PairMatrixAdd4(high, low, ih); PairMatrixAdd4(high, low, il);
        }
        gradients[candidate * leaves + row] = high.x + low.x;
        hessians[cell] = high[mass] + low[mass];
        if (!isfinite(gradients[candidate * leaves + row]))
            atomic_fetch_or_explicit(status + candidate, 4u, memory_order_relaxed);
    }
    if (!isfinite(hessians[cell])) atomic_fetch_or_explicit(status + candidate, 4u, memory_order_relaxed);
}
)METAL";
