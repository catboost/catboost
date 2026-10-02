#pragma once

// Port of libs/model/cuda/evaluator.cu::EvalObliviousTrees. A Metal thread
// evaluates one document and a tile of trees; a second dispatch combines tiles.
// The training border index is zero based: bin > split_bin is equivalent to
// CUDA evaluator's bin >= repacked FeatureVal (which is one based).
static const char* CBMInferenceSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct InferenceParams {
    uint rows, splitStride, leafStride, treeStart, treeEnd, tiles, treesPerTile;
    uint dimensions;
};

// Error-free two-sum followed by renormalization, with fast math disabled.
// Two float components preserve low-order leaf bits and cancellation that a
// single float accumulation loses; this is not a general IEEE float64 emulator.
inline float2 PairAdd(float2 accumulator, float2 value) {
    float sum = accumulator.x + value.x;
    float right = sum - accumulator.x;
    float error = (accumulator.x - (sum - right)) + (value.x - right);
    error += accumulator.y + value.y;
    float high = sum + error;
    return float2(high, error - (high - sum));
}

kernel void EvaluateTreeTiles(
    device const uchar* bins [[buffer(0)]],
    device const uint* depths [[buffer(1)]],
    device const uint* splitFeatures [[buffer(2)]],
    device const uint* splitBins [[buffer(3)]],
    device const float2* leafValues [[buffer(4)]],
    device float2* partials [[buffer(5)]],
    device const uchar* splitTypes [[buffer(6)]],
    constant InferenceParams& p [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]) {
    uint row = index.x;
    uint tile = index.y;
    uint dimension = index.z;
    if (row >= p.rows || tile >= p.tiles || dimension >= p.dimensions) return;
    uint begin = p.treeStart + tile * p.treesPerTile;
    uint end = min(begin + p.treesPerTile, p.treeEnd);
    float2 sum(0.0f);
    for (uint tree = begin; tree < end; ++tree) {
        uint leaf = 0;
        uint offset = tree * p.splitStride;
        for (uint level = 0; level < depths[tree]; ++level) {
            uint feature = splitFeatures[offset + level];
            uchar value = bins[feature * p.rows + row];
            uint border = splitBins[offset + level];
            bool right = splitTypes[offset + level] == 1 ? value == border : value > border;
            leaf |= uint(right) << level;
        }
        sum = PairAdd(sum, leafValues[(tree * p.leafStride + leaf) * p.dimensions + dimension]);
    }
    partials[(tile * p.rows + row) * p.dimensions + dimension] = sum;
}

kernel void ReduceTreeTiles(
    device const float2* partials [[buffer(0)]],
    device float2* predictions [[buffer(1)]],
    constant InferenceParams& p [[buffer(2)]],
    uint2 index [[thread_position_in_grid]]) {
    uint row = index.x, dimension = index.y;
    if (row >= p.rows || dimension >= p.dimensions) return;
    float2 sum(0.0f);
    for (uint tile = 0; tile < p.tiles; ++tile) {
        sum = PairAdd(sum, partials[(tile * p.rows + row) * p.dimensions + dimension]);
    }
    predictions[row * p.dimensions + dimension] = sum;
}

struct InferenceNode {
    uint feature, bin, type, left, right, leaf;
};

kernel void EvaluateNonSymmetricTreeTiles(
    device const uchar* bins [[buffer(0)]],
    device const uint* roots [[buffer(1)]],
    device const InferenceNode* nodes [[buffer(2)]],
    device const float2* leaves [[buffer(3)]],
    device float2* partials [[buffer(4)]],
    constant InferenceParams& p [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]) {
    const uint row = index.x, tile = index.y, dimension = index.z;
    if (row >= p.rows || tile >= p.tiles || dimension >= p.dimensions) return;
    const uint begin = p.treeStart + tile * p.treesPerTile;
    const uint end = min(begin + p.treesPerTile, p.treeEnd);
    float2 sum(0.0f);
    for (uint tree = begin; tree < end; ++tree) {
        uint current = roots[tree];
        // The host validates every graph, terminal reference and depth before
        // upload. No padded complete tree or recursive traversal is needed.
        while (nodes[current].leaf == 0xffffffffu) {
            const InferenceNode node = nodes[current];
            const uchar value = bins[node.feature * p.rows + row];
            const bool right = node.type == 1 ? value == node.bin : value > node.bin;
            current = right ? node.right : node.left;
        }
        sum = PairAdd(sum, leaves[nodes[current].leaf * p.dimensions + dimension]);
    }
    partials[(tile * p.rows + row) * p.dimensions + dimension] = sum;
}
)MSL";
