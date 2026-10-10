#pragma once

#include <stdint.h>

// Standalone grow-policy primitives. All layouts below have the same scalar
// alignment in C++ and MSL; no shared symmetric-tree KernelParams are required.
struct CBMGreedyParams {
    uint32_t Rows, Features, Leaves, Candidates;
    uint32_t TotalBins, ScoreGroups, ScoreFunction, MinDataInLeaf;
    uint32_t MaxDepth, MaxLeaves, Policy, Dimensions;
    uint32_t HistogramStride, LeafStride, MulticlassOptimization, Normalize;
    float L2;
    uint32_t FeatureBegin, Reserved0, Reserved1;
};
struct CBMGreedySplit {
    uint32_t Index, Feature, Bin, Type;
    float Gain;
    uint32_t Valid, Error, Leaf;
};
struct CBMGreedyFrontier {
    uint32_t Selected, NewLeaves, Error, Reserved;
};
static_assert(sizeof(CBMGreedyParams) == 80);
static_assert(sizeof(CBMGreedySplit) == 32);
static_assert(sizeof(CBMGreedyFrontier) == 16);

// Contract:
// * Policy: 0=Depthwise, 1=Lossguide, 2=Region. ScoreFunction: 0/2=L2,
//   1/3=Cosine, 4=SolarL2, 5=LOOL2, 6=SatL2. Only score IDs 2/3 use
//   Hessians; all other scores use observation weights. IDs4..6 ignore L2.
// * Compact histogram cell = leaf*TotalBins + feature_offsets[local feature]
//   + candidate bin. Candidate feature IDs are global; FeatureBegin/Features
//   describe the current feature tile. Numeric bins contain prefix statistics;
//   one-hot bins contain equality statistics. Types: 0=numeric, 1=one-hot.
// * Gradient planes use HistogramStride and LeafStride. Dimensions=1 is the
//   scalar case. MulticlassOptimization adds the negative-sum missing class.
//   Denominator statistics are one shared plane, as in CUDA's greedy search.
// * Find groups=(ScoreGroups,Leaves,1), Reduce groups=(Leaves,1,1), 256 threads
//   each. Find/Reduce output all leaves. Retained untouched-leaf winners can
//   instead be merged by the caller; independent feature tiles compare global
//   candidate indices to preserve ties. All nonfinite scoring errors propagate.
// * CUDA zero-child-weight candidates have gain=0. Lossguide can choose zero
//   or positive gains; Depthwise requires gain<0. Region requires gain<0 after
//   its root. Feature noise is applied to both scores before subtraction.
// * MinDataInLeaf is a PARENT row-count terminal check, not a minimum child
//   count or weight. CUDA's initial root is not marked terminal by this check.
// * Select uses one thread. Left child retains its parent leaf ID; right IDs
//   append in increasing parent-ID order. Region ties use smallest leaf ID,
//   making its otherwise unspecified equal-score sibling order deterministic.
// * Route updates row leaf IDs in place. Count scans appended-right-child flags
//   in 256-position tiles; Scan makes tile prefixes; Build derives arbitrary
//   leaf-count offsets; Scatter stably moves right children after retained left
//   children. Right IDs must append in parent-ID order. Count/Scan require 256
//   threads/group. Workspace is uint[rows] + uint[ceil(rows/256)]. Old/new index
//   and offset buffers MUST be distinct. Rows retain their previous stable order.
//   The resulting indices/offsets feed compact histograms with reuse=0; the
//   symmetric smaller-child reuse ABI must not be used with appended children.
// * Depthwise respects MaxLeaves by taking eligible leaves in leaf-ID order
//   if the remaining capacity is insufficient. Normal CUDA Depthwise options
//   allocate enough capacity for a whole level. The caller owns node/path
//   construction, leaf estimation, model export, and complete training.
static const char* CBMMetalGreedySource = R"METAL(
#include <metal_stdlib>
using namespace metal;

struct GreedyParams {
    uint rows, features, leaves, candidates;
    uint total_bins, score_groups, score_function, min_data_in_leaf;
    uint max_depth, max_leaves, policy, dimensions;
    uint histogram_stride, leaf_stride, multiclass_optimization, normalize;
    float l2;
    uint feature_begin, reserved0, reserved1;
};
struct GreedySplit {
    uint index, feature, bin, type;
    float gain;
    uint valid, error, leaf;
};
struct GreedyFrontier { uint selected, new_leaves, error, reserved; };

inline GreedySplit EmptyGreedySplit(uint leaf) {
    return {0xffffffffu, 0, 0, 0, INFINITY, 0, 0, leaf};
}
inline bool BetterGreedySplit(GreedySplit other, GreedySplit current) {
    return other.valid && (!current.valid || other.gain < current.gain ||
        (other.gain == current.gain && other.index < current.index));
}
inline void AddGreedyLeaf(float sum, float weight, constant GreedyParams& p,
                           thread float3& score) {
    if (p.score_function == 4) {
        if (weight > 1e-20f) score.x += (-sum / weight) * sum * (1.0f + 2.0f * log(weight + 1.0f));
    } else if (p.score_function == 5) {
        float adjust = weight > 1.0f ? weight / (weight - 1.0f) : 0.0f;
        adjust *= adjust;
        if (weight > 0.0f) score.x -= (sum * adjust) * (sum / weight);
    } else if (p.score_function == 6) {
        if (weight > 2.0f) {
            // Preserve CUDA's negative interval before (3+sqrt(5))/2.
            // Near that pole w-3 is exact and fma evaluates the polynomial
            // without cancellation from separately rounded products. Scaling
            // above4 avoids overflow in the otherwise quadratic formula.
            const float inverse = 1.0f / weight;
            const float adjust = weight <= 4.0f
                ? weight * (weight - 2.0f) / fma(weight, weight - 3.0f, 1.0f)
                : (1.0f - 2.0f * inverse) / (1.0f - 3.0f * inverse + inverse * inverse);
            score.x -= (sum * adjust) * (sum / weight);
        }
    } else if ((p.score_function & 1u) == 0) {
        if (weight > 1e-20f) score.x -= sum * (sum / (weight + p.l2));
    } else {
        const float lambda = p.normalize ? p.l2 * weight : p.l2;
        const float mu = weight > 0 ? sum / (weight + lambda) : 0;
        score.y += sum * mu;
        score.z += weight * mu * mu;
    }
}
inline float FinishGreedyScore(float3 score, float noise, constant GreedyParams& p) {
    return (p.score_function == 1 || p.score_function == 3) ? -score.y / sqrt(score.z) + noise : score.x;
}

// CUDA greedy_subsets_searcher/kernel/compute_scores.cu:
// ComputeOptimalSplitsRegion / ComputeOptimalSplit, with deterministic block
// reduction. Each candidate measures after-minus-before for ONE leaf.
kernel void FindGreedySplitWinners(
    const device float* sums [[buffer(0)]], const device float* weights [[buffer(1)]],
    const device float* leaf_sums [[buffer(2)]], const device float* leaf_weights [[buffer(3)]],
    const device uint* features [[buffer(4)]], const device uint* bins [[buffer(5)]],
    const device uchar* types [[buffer(6)]], const device uint* feature_offsets [[buffer(7)]],
    const device float* feature_weights [[buffer(8)]], const device float* feature_noise [[buffer(9)]],
    device GreedySplit* winners [[buffer(10)]], constant GreedyParams& p [[buffer(11)]],
    uint2 local_thread [[thread_position_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint tid = local_thread.x, leaf = group.y;
    GreedySplit best = EmptyGreedySplit(leaf);
    for (uint candidate = group.x * 256 + tid; candidate < p.candidates;
         candidate += p.score_groups * 256) {
        const uint feature = features[candidate];
        if (feature < p.feature_begin || feature - p.feature_begin >= p.features) continue;
        const uint local = feature - p.feature_begin;
        const ulong cell = ulong(leaf) * p.total_bins + feature_offsets[local] + bins[candidate];
        const float parent_weight = leaf_weights[leaf];
        const float selected_weight = max(weights[cell], 0.0f);
        const float other_weight = max(parent_weight - selected_weight, 0.0f);
        const bool zero_child = selected_weight < 1e-20f || other_weight < 1e-20f;
        float3 after = float3(0, 0, 1e-10f), before = after;
        float missing_selected = 0, missing_parent = 0;
        for (uint k = 0; k < p.dimensions + p.multiclass_optimization; ++k) {
            float selected, parent;
            if (k < p.dimensions) {
                selected = sums[ulong(k) * p.histogram_stride + cell];
                parent = leaf_sums[ulong(k) * p.leaf_stride + leaf];
                missing_selected -= selected;
                missing_parent -= parent;
            } else {
                selected = missing_selected;
                parent = missing_parent;
            }
            // Equality versus prefix routing swaps sides but not their score.
            AddGreedyLeaf(selected, selected_weight, p, after);
            AddGreedyLeaf(parent - selected, other_weight, p, after);
            AddGreedyLeaf(parent, parent_weight, p, before);
        }
        float gain = zero_child ? 0.0f :
            FinishGreedyScore(after, feature_noise[feature], p) -
            FinishGreedyScore(before, feature_noise[feature], p);
        gain *= feature_weights[feature];
        best.error |= uint(!isfinite(gain));
        GreedySplit value = {candidate, feature, bins[candidate], uint(types[candidate]),
            gain, uint(isfinite(gain)), 0, leaf};
        if (BetterGreedySplit(value, best)) {
            const uint error = best.error;
            best = value;
            best.error = error;
        }
    }
    threadgroup GreedySplit values[256];
    values[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const GreedySplit other = values[tid + stride];
            const uint error = values[tid].error | other.error;
            if (BetterGreedySplit(other, values[tid])) values[tid] = other;
            values[tid].error = error;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) winners[leaf * p.score_groups + group.x] = values[0];
}

kernel void ReduceGreedySplitWinners(
    const device GreedySplit* partials [[buffer(0)]], device GreedySplit* winners [[buffer(1)]],
    constant GreedyParams& p [[buffer(2)]], uint tid [[thread_position_in_threadgroup]],
    uint leaf [[threadgroup_position_in_grid]]) {
    threadgroup GreedySplit values[256];
    GreedySplit best = EmptyGreedySplit(leaf);
    for (uint block = tid; block < p.score_groups; block += 256) {
        const GreedySplit other = partials[leaf * p.score_groups + block];
        const uint error = best.error | other.error;
        if (BetterGreedySplit(other, best)) best = other;
        best.error = error;
    }
    values[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) {
            const GreedySplit other = values[tid + stride];
            const uint error = values[tid].error | other.error;
            if (BetterGreedySplit(other, values[tid])) values[tid] = other;
            values[tid].error = error;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) winners[leaf] = values[0];
}

// Host CUDA scheduling translated to a small device frontier operation.
// Recompute/retain every leaf winner before this call. No histogram access.
kernel void SelectGreedyLeaves(
    const device GreedySplit* winners [[buffer(0)]], const device uint* offsets [[buffer(1)]],
    const device uint* depths [[buffer(2)]], device uint* selected [[buffer(3)]],
    device uint* right_ids [[buffer(4)]], device GreedyFrontier* info [[buffer(5)]],
    constant GreedyParams& p [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    GreedyFrontier result = {0, p.leaves, 0, 0};
    uint max_depth = 0;
    for (uint leaf = 0; leaf < p.leaves; ++leaf) {
        right_ids[leaf] = 0xffffffffu;
        max_depth = max(max_depth, depths[leaf]);
        result.error |= winners[leaf].error;
    }
    if (result.error || p.leaves >= p.max_leaves) { info[0] = result; return; }
    uint best_leaf = 0xffffffffu;
    float best_gain = INFINITY;
    for (uint leaf = 0; leaf < p.leaves; ++leaf) {
        const bool root = p.leaves == 1 && depths[leaf] == 0;
        if (!winners[leaf].valid || depths[leaf] >= p.max_depth ||
            (!root && offsets[leaf + 1] - offsets[leaf] <= p.min_data_in_leaf)) continue;
        const float gain = winners[leaf].gain;
        if (p.policy == 0) {
            if (gain < 0 && result.new_leaves < p.max_leaves) {
                selected[result.selected++] = leaf;
                right_ids[leaf] = result.new_leaves++;
            }
        } else {
            if (p.policy == 2 && (depths[leaf] != max_depth || (!root && gain >= 0))) continue;
            if (gain < best_gain) { best_gain = gain; best_leaf = leaf; }
        }
    }
    if (p.policy != 0 && best_leaf != 0xffffffffu) {
        selected[result.selected++] = best_leaf;
        right_ids[best_leaf] = result.new_leaves++;
    }
    info[0] = result;
}

kernel void RouteGreedySplitRows(
    const device uchar* bins [[buffer(0)]], device uint* leaf_ids [[buffer(1)]],
    const device GreedySplit* winners [[buffer(2)]], const device uint* right_ids [[buffer(3)]],
    constant GreedyParams& p [[buffer(4)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    const uint parent = leaf_ids[row], right_id = right_ids[parent];
    if (right_id == 0xffffffffu) return;
    const GreedySplit split = winners[parent];
    const uint value = uint(bins[ulong(split.feature) * p.rows + row]);
    const bool right = split.type ? value == split.bin : value > split.bin;
    if (right) leaf_ids[row] = right_id;
}

kernel void UpdateGreedyLeafDepths(
    const device uint* old_depths [[buffer(0)]], const device uint* right_ids [[buffer(1)]],
    device uint* new_depths [[buffer(2)]], constant GreedyParams& p [[buffer(3)]],
    uint leaf [[thread_position_in_grid]]) {
    if (leaf >= p.leaves) return;
    const uint right = right_ids[leaf];
    const uint depth = old_depths[leaf] + uint(right != 0xffffffffu);
    new_depths[leaf] = depth;
    if (right != 0xffffffffu) new_depths[right] = depth;
}

// Stable global flag partition, equivalent to CUDA's per-leaf sequence/sort
// because old indices are parent ordered and right IDs append in that order.
// Groups: ceil(rows/256), 256 threads. Independent tiles expose root parallelism.
kernel void CountGreedyPartitionBits(
    const device uint* old_indices [[buffer(0)]], const device uint* leaf_ids [[buffer(1)]],
    device uint* row_prefix [[buffer(2)]], device uint* tile_prefix [[buffer(3)]],
    constant GreedyParams& p [[buffer(4)]], uint tid [[thread_position_in_threadgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
    threadgroup uint prefix[256];
    const uint position = tile * 256 + tid;
    prefix[tid] = position < p.rows ? uint(leaf_ids[old_indices[position]] >= p.leaves) : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        const uint add = tid >= stride ? prefix[tid - stride] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (position < p.rows) row_prefix[position] = prefix[tid];
    if (tid == 255) tile_prefix[tile] = prefix[255];
}

// One group of 256 threads. Only ceil(rows/256) totals are scanned here; input
// row scans and final scatter are parallel even when only the root is active.
kernel void ScanGreedyPartitionTiles(
    device uint* tile_prefix [[buffer(0)]], constant GreedyParams& p [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint prefix[256];
    threadgroup uint chunk_total;
    uint carry = 0;
    const uint tiles = (p.rows - 1) / 256 + 1;
    for (uint start = 0; start < tiles; start += 256) {
        const uint tile = start + tid;
        prefix[tid] = tile < tiles ? tile_prefix[tile] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 1; stride < 256; stride <<= 1) {
            const uint add = tid >= stride ? prefix[tid - stride] : 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            prefix[tid] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tile < tiles) tile_prefix[tile] = carry + prefix[tid];
        if (tid == 255) chunk_total = prefix[255];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        carry += chunk_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

inline uint GreedyRightBefore(uint position, const device uint* row_prefix,
                              const device uint* tile_prefix) {
    if (position == 0) return 0;
    const uint last = position - 1, tile = last / 256;
    return row_prefix[last] + (tile ? tile_prefix[tile - 1] : 0);
}

// Threads: old leaf count. Input/output offsets must be distinct. Empty
// parents/children have repeated offsets, including those at either endpoint.
kernel void BuildGreedyPartitionOffsets(
    const device uint* old_offsets [[buffer(0)]], const device uint* right_ids [[buffer(1)]],
    const device uint* row_prefix [[buffer(2)]], const device uint* tile_prefix [[buffer(3)]],
    const device GreedyFrontier* info [[buffer(4)]], device uint* offsets [[buffer(5)]],
    constant GreedyParams& p [[buffer(6)]], uint parent [[thread_position_in_grid]]) {
    if (parent >= p.leaves) return;
    const uint start = old_offsets[parent];
    const uint before = GreedyRightBefore(start, row_prefix, tile_prefix);
    offsets[parent] = start - before;
    const uint right_id = right_ids[parent];
    if (right_id != 0xffffffffu) {
        const uint total_left = p.rows - tile_prefix[(p.rows - 1) / 256];
        offsets[right_id] = total_left + before;
    }
    if (parent == 0) offsets[info[0].new_leaves] = p.rows;
}

kernel void ScatterGreedyPartitionRows(
    const device uint* old_indices [[buffer(0)]], const device uint* leaf_ids [[buffer(1)]],
    const device uint* row_prefix [[buffer(2)]], const device uint* tile_prefix [[buffer(3)]],
    device uint* new_indices [[buffer(4)]], constant GreedyParams& p [[buffer(5)]],
    uint position [[thread_position_in_grid]]) {
    if (position < p.rows) {
        const uint row = old_indices[position];
        const uint through = GreedyRightBefore(position + 1, row_prefix, tile_prefix);
        const uint total_left = p.rows - tile_prefix[(p.rows - 1) / 256];
        const uint destination = leaf_ids[row] >= p.leaves
            ? total_left + through - 1 : position - through;
        new_indices[destination] = row;
    }
}
)METAL";
