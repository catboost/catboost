// scores.metal — Split score computation kernel for CatBoost-MLX.
// Translates catboost/cuda/methods/greedy_subsets_searcher/kernel/compute_scores.cu to Metal.
//
// Given histograms and partition statistics, compute the L2 split score for each
// (feature, bin) candidate and find the best split per partition pair.
//
// L2 score for a split:
//   score = -sumLeft^2 / (weightLeft + lambda) - sumRight^2 / (weightRight + lambda)
// The split that minimizes score (most negative) is the best.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint SIMD_SIZE = 32;

// Per-feature, per-bin score computation.
// Each thread evaluates one bin-feature candidate.
//
// Grid:   (ceil(totalBinFeatures / BLOCK_SIZE), numPartitionPairs, 1)
// Thread: (BLOCK_SIZE, 1, 1) where BLOCK_SIZE = 256
kernel void compute_split_scores(
    // Histogram: [numPartitions, numStats, totalBinFeatures]
    const device float*      histogram        [[buffer(0)]],
    // Partition statistics: [numPartitions] with {weight, sum, count}
    const device float*      partWeights      [[buffer(1)]],
    const device float*      partSums         [[buffer(2)]],
    // Feature metadata: firstFoldIndex and folds per feature
    const device uint*       featureFirstFold [[buffer(3)]],  // [numFeatures]
    const device uint*       featureFolds     [[buffer(4)]],  // [numFeatures]
    constant uint&           numFeatures      [[buffer(5)]],
    // Configuration
    constant uint&           totalBinFeatures [[buffer(6)]],
    constant uint&           numStats         [[buffer(7)]],
    constant float&          l2RegLambda      [[buffer(8)]],
    // Partition pair: which left/right partition to evaluate
    const device uint*       leftPartIds      [[buffer(9)]],   // [numPairs]
    const device uint*       rightPartIds     [[buffer(10)]],  // [numPairs]
    // Output: best split per block per pair — [numPairs * numBlocks]
    // CPU-side reduction picks the global best across blocks.
    device float*            bestScores       [[buffer(11)]],  // [numPairs * numBlocks]
    device uint*             bestFeatureIds   [[buffer(12)]],  // [numPairs * numBlocks]
    device uint*             bestBinIds       [[buffer(13)]],  // [numPairs * numBlocks]

    uint3 threadgroup_position_in_grid   [[threadgroup_position_in_grid]],
    uint3 threadgroups_per_grid          [[threadgroups_per_grid]],
    uint  thread_index_in_threadgroup    [[thread_index_in_threadgroup]]
) {
    constexpr uint BLOCK_SIZE = 256;

    const uint pairIdx = threadgroup_position_in_grid.y;
    const uint binFeatureIdx = threadgroup_position_in_grid.x * BLOCK_SIZE + thread_index_in_threadgroup;

    const uint leftPartId = leftPartIds[pairIdx];
    const uint rightPartId = rightPartIds[pairIdx];

    // Load partition statistics
    const float totalWeight = partWeights[leftPartId] + partWeights[rightPartId];
    const float totalSum = partSums[leftPartId] + partSums[rightPartId];

    // Shared memory for block-level argmax reduction
    threadgroup float  sharedScores[BLOCK_SIZE];
    threadgroup uint   sharedFeatures[BLOCK_SIZE];
    threadgroup uint   sharedBins[BLOCK_SIZE];

    float myScore = INFINITY;
    uint myFeatureId = 0xFFFFFFFF;
    uint myBinId = 0;

    if (binFeatureIdx < totalBinFeatures) {
        // Find which feature this bin belongs to
        uint featureId = 0;
        uint binInFeature = binFeatureIdx;
        for (uint f = 0; f < numFeatures; f++) {
            if (binInFeature < featureFolds[f]) {
                featureId = f;
                break;
            }
            binInFeature -= featureFolds[f];
        }

        // Read histogram values for left partition
        // Gradient histogram (stat=0)
        const uint histBase = leftPartId * numStats * totalBinFeatures;
        const float sumLeft = histogram[histBase + binFeatureIdx];

        // Weight histogram (stat=1), or use count if no weights
        float weightLeft;
        if (numStats > 1) {
            weightLeft = histogram[histBase + totalBinFeatures + binFeatureIdx];
        } else {
            // Without separate weight hist, approximate from partition stats
            // (will be refined when weight histograms are computed)
            weightLeft = partWeights[leftPartId];
        }

        const float sumRight = totalSum - sumLeft;
        const float weightRight = totalWeight - weightLeft;

        // L2 score: -sum^2 / (weight + lambda)
        const float lambda = l2RegLambda;
        float score = 0.0f;

        if (weightLeft > 1e-15f) {
            score -= (sumLeft * sumLeft) / (weightLeft + lambda);
        }
        if (weightRight > 1e-15f) {
            score -= (sumRight * sumRight) / (weightRight + lambda);
        }

        myScore = score;
        myFeatureId = featureId;
        myBinId = binInFeature;
    }

    // Block-level argmax reduction
    sharedScores[thread_index_in_threadgroup] = myScore;
    sharedFeatures[thread_index_in_threadgroup] = myFeatureId;
    sharedBins[thread_index_in_threadgroup] = myBinId;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for minimum score
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (thread_index_in_threadgroup < stride) {
            const uint other = thread_index_in_threadgroup + stride;
            if (sharedScores[other] < sharedScores[thread_index_in_threadgroup]) {
                sharedScores[thread_index_in_threadgroup] = sharedScores[other];
                sharedFeatures[thread_index_in_threadgroup] = sharedFeatures[other];
                sharedBins[thread_index_in_threadgroup] = sharedBins[other];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Thread 0 writes per-block results. The CPU-side reduction finds the global best.
    // Each block writes to its own slot: [pairIdx * numBlocks + blockIdx]
    // to avoid race conditions between threadgroups.
    if (thread_index_in_threadgroup == 0) {
        const uint blockIdx = threadgroup_position_in_grid.x;
        const uint numBlocks = threadgroups_per_grid.x;
        const uint outIdx = pairIdx * numBlocks + blockIdx;
        bestScores[outIdx] = sharedScores[0];
        bestFeatureIds[outIdx] = sharedFeatures[0];
        bestBinIds[outIdx] = sharedBins[0];
    }
}
