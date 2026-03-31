// leaves.metal — Leaf value estimation and tree application kernels for CatBoost-MLX.
//
// Two kernels:
//   1. compute_leaf_values: Newton step to compute optimal leaf predictions
//   2. apply_oblivious_tree: Evaluate tree on all docs, update prediction cursor

#include <metal_stdlib>
using namespace metal;

// Compute leaf values using Newton's method.
// leaf_value = -gradient_sum / (hessian_sum + lambda)
//
// Grid: (numLeaves, 1, 1)
// Thread: (1, 1, 1) — one thread per leaf (leaves are few, ~2-256)
kernel void compute_leaf_values(
    const device float*  gradientSums   [[buffer(0)]],  // [numLeaves]
    const device float*  hessianSums    [[buffer(1)]],  // [numLeaves]
    constant float&      l2RegLambda    [[buffer(2)]],
    constant float&      learningRate   [[buffer(3)]],
    constant uint&       numLeaves      [[buffer(4)]],
    device float*        leafValues     [[buffer(5)]],  // [numLeaves] output

    uint thread_position_in_grid [[thread_position_in_grid]]
) {
    if (thread_position_in_grid >= numLeaves) return;

    const uint leafIdx = thread_position_in_grid;
    const float gradSum = gradientSums[leafIdx];
    const float hessSum = hessianSums[leafIdx];

    // Newton step with L2 regularization
    float value = 0.0f;
    if (abs(hessSum + l2RegLambda) > 1e-20f) {
        value = -gradSum / (hessSum + l2RegLambda);
    }

    // Scale by learning rate
    leafValues[leafIdx] = learningRate * value;
}

// Apply an oblivious (symmetric) tree to all documents, updating the prediction cursor.
//
// For a depth-d oblivious tree:
//   leaf_idx = 0
//   for level in 0..d-1:
//       leaf_idx |= (feature_value[doc][splitFeature[level]] > splitBin[level]) << level
//   cursor[doc] += leafValues[leaf_idx]
//
// Grid: (ceil(numDocs / BLOCK_SIZE), 1, 1)
// Thread: (BLOCK_SIZE, 1, 1)
kernel void apply_oblivious_tree(
    // Compressed feature index: [numDocs, numUi32PerDoc]
    const device uint*   compressedIndex  [[buffer(0)]],
    constant uint&       lineSize         [[buffer(1)]],  // numUi32PerDoc
    // Tree structure: one split per depth level
    const device uint*   splitFeatureCol  [[buffer(2)]],  // [depth] — which ui32 column
    const device uint*   splitShift       [[buffer(3)]],  // [depth] — bit shift
    const device uint*   splitMask        [[buffer(4)]],  // [depth] — post-shift mask (0xFF for 1-byte)
    const device uint*   splitBin         [[buffer(5)]],  // [depth] — threshold bin
    constant uint&       treeDepth        [[buffer(6)]],
    // Leaf values
    const device float*  leafValues       [[buffer(7)]],  // [2^depth]
    // Prediction cursor to update (in-place)
    device float*        cursor           [[buffer(8)]],  // [numDocs]
    constant uint&       numDocs          [[buffer(9)]],
    // Document-to-partition mapping (optional, for partition update)
    device uint*         partitions       [[buffer(10)]],  // [numDocs] — updated leaf assignments

    uint thread_position_in_grid [[thread_position_in_grid]]
) {
    if (thread_position_in_grid >= numDocs) return;

    const uint docIdx = thread_position_in_grid;

    // Compute leaf index by evaluating all split conditions
    uint leafIdx = 0;
    for (uint level = 0; level < treeDepth; level++) {
        const uint col = splitFeatureCol[level];
        const uint shift = splitShift[level];
        const uint mask = splitMask[level];
        const uint threshold = splitBin[level];

        const uint packed = compressedIndex[docIdx * lineSize + col];
        const uint featureValue = (packed >> shift) & mask;

        // If feature value > threshold, go right (set bit)
        if (featureValue > threshold) {
            leafIdx |= (1u << level);
        }
    }

    // Update cursor with leaf value
    cursor[docIdx] += leafValues[leafIdx];

    // Update partition (leaf) assignment for next iteration
    partitions[docIdx] = leafIdx;
}
