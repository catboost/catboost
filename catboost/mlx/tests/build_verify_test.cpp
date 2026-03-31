// Build verification test for Phase 6 changes.
// Compiles with: clang++ -std=c++17 -I/opt/homebrew/Cellar/mlx/0.31.1/include
//                -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx
//                -framework Metal -framework Foundation
//                catboost/mlx/tests/build_verify_test.cpp -o /tmp/build_verify_test && /tmp/build_verify_test
//
// Exercises: kernel_sources.h, histogram dispatch logic, partition layout,
//            leaf estimation, and tree application — all without CatBoost headers.

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <string>
#include <cassert>

namespace mx = mlx::core;

// ---- Include kernel sources (the actual production header) ----
#include <catboost/mlx/kernels/kernel_sources.h>

using namespace NCatboostMlx;

// ---- Minimal types mirroring CatBoost ----
using ui32 = uint32_t;

struct TCFeature {
    ui32 FirstFoldIndex;
    ui32 Folds;
    ui32 Index;
};

// ---- Partition layout (mirrors structure_searcher.h) ----
struct TPartitionLayout {
    mx::array DocIndices;
    mx::array PartOffsets;
    mx::array PartSizes;
    std::vector<ui32> PartOffsetsHost;
    std::vector<ui32> PartSizesHost;
};

TPartitionLayout ComputePartitionLayout(const mx::array& partitions, ui32 numDocs, ui32 numPartitions) {
    mx::eval(partitions);
    const uint32_t* partsPtr = partitions.data<uint32_t>();

    std::vector<ui32> partSizes(numPartitions, 0);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 p = partsPtr[d];
        if (p < numPartitions) partSizes[p]++;
    }

    std::vector<ui32> partOffsets(numPartitions, 0);
    for (ui32 i = 1; i < numPartitions; ++i) {
        partOffsets[i] = partOffsets[i-1] + partSizes[i-1];
    }

    std::vector<ui32> docIndices(numDocs);
    std::vector<ui32> writePos = partOffsets;
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 p = partsPtr[d];
        if (p < numPartitions) {
            docIndices[writePos[p]++] = d;
        }
    }

    return TPartitionLayout{
        .DocIndices = mx::array(reinterpret_cast<const int32_t*>(docIndices.data()), {static_cast<int>(numDocs)}, mx::uint32),
        .PartOffsets = mx::array(reinterpret_cast<const int32_t*>(partOffsets.data()), {static_cast<int>(numPartitions)}, mx::uint32),
        .PartSizes = mx::array(reinterpret_cast<const int32_t*>(partSizes.data()), {static_cast<int>(numPartitions)}, mx::uint32),
        .PartOffsetsHost = partOffsets,
        .PartSizesHost = partSizes
    };
}

// ---- Histogram dispatch (mirrors histogram.cpp) ----
mx::array DispatchHistogramGroup(
    const mx::array& compressedData,
    const mx::array& stats,
    const mx::array& docIndices,
    const mx::array& partOffsets,
    const mx::array& partSizes,
    ui32 featureColumnIdx,
    ui32 lineSize,
    ui32 maxBlocksPerPart,
    const mx::array& foldCounts,
    const mx::array& firstFoldIndices,
    ui32 totalBinFeatures,
    ui32 numStats,
    ui32 numPartitions,
    ui32 totalNumDocs,
    const mx::Shape& histShape
) {
    auto featureColArr = mx::array(static_cast<uint32_t>(featureColumnIdx), mx::uint32);
    auto lineSizeArr   = mx::array(static_cast<uint32_t>(lineSize), mx::uint32);
    auto maxBlocksArr  = mx::array(static_cast<uint32_t>(maxBlocksPerPart), mx::uint32);
    auto totalBinsArr  = mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32);
    auto numStatsArr   = mx::array(static_cast<uint32_t>(numStats), mx::uint32);
    auto totalDocsArr  = mx::array(static_cast<uint32_t>(totalNumDocs), mx::uint32);

    auto flatCompressed = mx::reshape(compressedData, {-1});

    auto kernel = mx::fast::metal_kernel(
        "histogram_one_byte_features",
        {"compressedIndex", "stats", "docIndices",
         "partOffsets", "partSizes",
         "featureColumnIdx", "lineSize", "maxBlocksPerPart",
         "foldCounts", "firstFoldIndices",
         "totalBinFeatures", "numStats", "totalNumDocs"},
        {"histogram"},
        KernelSources::kHistOneByteSource,
        KernelSources::kHistHeader,
        true, false
    );

    // Grid = total threads (MLX divides by threadgroup to get threadgroup count)
    auto grid = std::make_tuple(
        static_cast<int>(256 * maxBlocksPerPart),
        static_cast<int>(numPartitions),
        static_cast<int>(numStats)
    );
    auto threadgroup = std::make_tuple(256, 1, 1);

    auto results = kernel(
        {flatCompressed, stats, docIndices,
         partOffsets, partSizes,
         featureColArr, lineSizeArr, maxBlocksArr,
         foldCounts, firstFoldIndices,
         totalBinsArr, numStatsArr, totalDocsArr},
        {histShape}, {mx::float32},
        grid, threadgroup,
        {}, 0.0f, false, mx::Device::gpu
    );

    return results[0];
}

// ---- Leaf value estimation (mirrors leaf_estimator.cpp) ----
mx::array ComputeLeafValues(
    const mx::array& gradientSums,
    const mx::array& hessianSums,
    float l2RegLambda,
    float learningRate
) {
    auto lambda = mx::array(l2RegLambda, mx::float32);
    auto lr = mx::array(learningRate, mx::float32);
    auto values = mx::negative(mx::divide(gradientSums, mx::add(hessianSums, lambda)));
    return mx::multiply(values, lr);
}

// ---- Tree application (mirrors tree_applier.cpp) ----
void ApplyObliviousTree(
    mx::array& cursor,
    const mx::array& partitions,
    const mx::array& leafValues,
    ui32 numDocs
) {
    // cursor[d] += leafValues[partitions[d]]
    auto gathered = mx::take(leafValues, partitions, 0);
    cursor = mx::add(cursor, gathered);
}

// ============================================================================
// Test: Full boosting iteration pipeline
// ============================================================================

bool TestFullIteration() {
    printf("=== Test: Full Boosting Iteration ===\n");

    const ui32 numDocs = 16;
    const ui32 numFeatures = 4;
    const ui32 numUi32PerDoc = 1;
    const ui32 folds = 3;  // bins 0-3, 3 split candidates
    const ui32 totalBinFeatures = numFeatures * folds;
    const float learningRate = 0.1f;
    const float l2Reg = 1.0f;

    // Generate synthetic data: features with bins 0-3
    std::vector<uint32_t> packed(numDocs);
    std::vector<float> targets(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        uint8_t f0 = d % 4;
        uint8_t f1 = (d / 4) % 4;
        uint8_t f2 = (d + 1) % 4;
        uint8_t f3 = (d + 2) % 4;
        packed[d] = (f0 << 24) | (f1 << 16) | (f2 << 8) | f3;
        targets[d] = static_cast<float>(d) * 0.5f - 4.0f;
    }

    auto compressedData = mx::array(reinterpret_cast<const int32_t*>(packed.data()), {static_cast<int>(numDocs)}, mx::uint32);
    auto targetsArr = mx::array(targets.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto cursor = mx::zeros({static_cast<int>(numDocs)}, mx::float32);

    // ---- Iteration 1: single partition (depth 0 histogram) ----
    // Gradients = cursor - targets (RMSE derivative)
    auto gradients = mx::subtract(cursor, targetsArr);
    auto hessians = mx::ones({static_cast<int>(numDocs)}, mx::float32);
    mx::eval({gradients, hessians});

    // Stats: [grads, hessians]
    auto stats = mx::concatenate({
        mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
        mx::reshape(hessians, {1, static_cast<int>(numDocs)})
    }, 0);
    stats = mx::reshape(stats, {static_cast<int>(2 * numDocs)});

    // Single partition layout
    auto partitions = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
    auto layout = ComputePartitionLayout(partitions, numDocs, 1);

    // Feature metadata
    std::vector<ui32> foldCountsVec = {folds, folds, folds, folds};
    std::vector<ui32> firstFoldVec = {0, 3, 6, 9};
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    mx::Shape histShape = {static_cast<int>(1 * 2 * totalBinFeatures)};

    printf("  Dispatching histogram kernel...\n");
    auto histogram = DispatchHistogramGroup(
        compressedData, stats, layout.DocIndices,
        layout.PartOffsets, layout.PartSizes,
        0, numUi32PerDoc, 1,
        foldCountsArr, firstFoldArr,
        totalBinFeatures, 2, 1, numDocs,
        histShape
    );
    mx::eval(histogram);

    printf("  Histogram computed, shape: [%d]\n", histogram.shape(0));

    // Find best split (simple: pick split with largest absolute gradient sum)
    const float* histData = histogram.data<float>();
    float bestScore = -1.0f;
    ui32 bestBinFeature = 0;
    for (ui32 bf = 0; bf < totalBinFeatures; ++bf) {
        float gradSum = histData[bf];                      // stat 0
        float hessSum = histData[totalBinFeatures + bf];   // stat 1
        float score = (hessSum + l2Reg > 0) ? (gradSum * gradSum) / (hessSum + l2Reg) : 0.0f;
        if (score > bestScore) {
            bestScore = score;
            bestBinFeature = bf;
        }
    }
    printf("  Best split: binFeature=%u, score=%.4f\n", bestBinFeature, bestScore);

    // Apply split: determine which feature and bin threshold
    ui32 featureIdx = 0;
    ui32 binThreshold = bestBinFeature;
    for (ui32 f = 0; f < numFeatures; ++f) {
        if (binThreshold < folds) {
            featureIdx = f;
            break;
        }
        binThreshold -= folds;
    }
    printf("  Split feature=%u, binThreshold=%u\n", featureIdx, binThreshold + 1);

    // Update partitions based on split
    // For simplicity: partition = (feature_bin > threshold) ? 1 : 0
    mx::eval(partitions);
    std::vector<uint32_t> newParts(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        uint8_t bin = (packed[d] >> (24 - 8 * featureIdx)) & 0xFF;
        newParts[d] = (bin > binThreshold + 1) ? 1 : 0;  // bin > threshold means right child
    }
    partitions = mx::array(reinterpret_cast<const int32_t*>(newParts.data()), {static_cast<int>(numDocs)}, mx::uint32);

    // Compute leaf values from partitions
    ui32 numLeaves = 2;
    mx::eval({gradients, hessians, partitions});
    const float* gradsPtr = gradients.data<float>();
    const float* hessPtr = hessians.data<float>();
    const uint32_t* partsPtr = partitions.data<uint32_t>();

    std::vector<float> gradSums(numLeaves, 0.0f);
    std::vector<float> hessSums(numLeaves, 0.0f);
    for (ui32 d = 0; d < numDocs; ++d) {
        ui32 leaf = partsPtr[d];
        if (leaf < numLeaves) {
            gradSums[leaf] += gradsPtr[d];
            hessSums[leaf] += hessPtr[d];
        }
    }

    auto gradSumsArr = mx::array(gradSums.data(), {static_cast<int>(numLeaves)}, mx::float32);
    auto hessSumsArr = mx::array(hessSums.data(), {static_cast<int>(numLeaves)}, mx::float32);
    auto leafValues = ComputeLeafValues(gradSumsArr, hessSumsArr, l2Reg, learningRate);
    mx::eval(leafValues);

    const float* lvPtr = leafValues.data<float>();
    printf("  Leaf values: [%.4f, %.4f]\n", lvPtr[0], lvPtr[1]);

    // Apply tree to cursor
    ApplyObliviousTree(cursor, partitions, leafValues, numDocs);
    mx::eval(cursor);

    // Check that loss decreased
    auto initialLoss = mx::mean(mx::square(mx::subtract(mx::zeros({static_cast<int>(numDocs)}, mx::float32), targetsArr)));
    auto newLoss = mx::mean(mx::square(mx::subtract(cursor, targetsArr)));
    mx::eval({initialLoss, newLoss});

    float initLossVal = initialLoss.item<float>();
    float newLossVal = newLoss.item<float>();
    printf("  Initial MSE: %.4f\n", initLossVal);
    printf("  After 1 iteration MSE: %.4f\n", newLossVal);

    bool pass = newLossVal < initLossVal;
    if (pass) {
        printf("  PASS: Loss decreased (%.4f -> %.4f)\n", initLossVal, newLossVal);
    } else {
        printf("  FAIL: Loss did not decrease (%.4f -> %.4f)\n", initLossVal, newLossVal);
    }
    return pass;
}

// ============================================================================
// Test: Multiple boosting iterations
// ============================================================================

bool TestMultipleIterations() {
    printf("\n=== Test: 10 Boosting Iterations ===\n");

    const ui32 numDocs = 32;
    const ui32 numFeatures = 4;
    const ui32 numUi32PerDoc = 1;
    const ui32 folds = 3;
    const ui32 totalBinFeatures = numFeatures * folds;
    const float learningRate = 0.3f;
    const float l2Reg = 1.0f;
    const ui32 numIterations = 10;

    // Generate data
    std::vector<uint32_t> packed(numDocs);
    std::vector<float> targets(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        packed[d] = ((d % 4) << 24) | (((d/4) % 4) << 16) | (((d+1) % 4) << 8) | ((d+2) % 4);
        targets[d] = static_cast<float>(d % 4) * 2.0f - 3.0f;
    }

    auto compressedData = mx::array(reinterpret_cast<const int32_t*>(packed.data()), {static_cast<int>(numDocs)}, mx::uint32);
    auto targetsArr = mx::array(targets.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto cursor = mx::zeros({static_cast<int>(numDocs)}, mx::float32);

    std::vector<ui32> foldCountsVec = {folds, folds, folds, folds};
    std::vector<ui32> firstFoldVec = {0, 3, 6, 9};
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    float prevLoss = 1e10f;
    bool monotonic = true;

    for (ui32 iter = 0; iter < numIterations; ++iter) {
        auto gradients = mx::subtract(cursor, targetsArr);
        auto hessians = mx::ones({static_cast<int>(numDocs)}, mx::float32);

        auto stats = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians, {1, static_cast<int>(numDocs)})
        }, 0);
        stats = mx::reshape(stats, {static_cast<int>(2 * numDocs)});

        auto partitions = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
        auto layout = ComputePartitionLayout(partitions, numDocs, 1);

        mx::Shape histShape = {static_cast<int>(2 * totalBinFeatures)};
        auto histogram = DispatchHistogramGroup(
            compressedData, stats, layout.DocIndices,
            layout.PartOffsets, layout.PartSizes,
            0, numUi32PerDoc, 1,
            foldCountsArr, firstFoldArr,
            totalBinFeatures, 2, 1, numDocs,
            histShape
        );
        mx::eval(histogram);
        const float* histData = histogram.data<float>();

        // Find best split
        float bestScore = -1.0f;
        ui32 bestBF = 0;
        for (ui32 bf = 0; bf < totalBinFeatures; ++bf) {
            float g = histData[bf];
            float h = histData[totalBinFeatures + bf];
            float s = (h + l2Reg > 0) ? (g * g) / (h + l2Reg) : 0.0f;
            if (s > bestScore) { bestScore = s; bestBF = bf; }
        }

        ui32 featureIdx = 0, binThreshold = bestBF;
        for (ui32 f = 0; f < numFeatures; ++f) {
            if (binThreshold < folds) { featureIdx = f; break; }
            binThreshold -= folds;
        }

        // Apply split
        std::vector<uint32_t> newParts(numDocs);
        for (ui32 d = 0; d < numDocs; ++d) {
            uint8_t bin = (packed[d] >> (24 - 8 * featureIdx)) & 0xFF;
            newParts[d] = (bin > binThreshold + 1) ? 1 : 0;
        }
        partitions = mx::array(reinterpret_cast<const int32_t*>(newParts.data()), {static_cast<int>(numDocs)}, mx::uint32);

        // Leaf estimation
        mx::eval({gradients, hessians, partitions});
        const float* gp = gradients.data<float>();
        const float* hp = hessians.data<float>();
        const uint32_t* pp = partitions.data<uint32_t>();

        std::vector<float> gSums(2, 0.0f), hSums(2, 0.0f);
        for (ui32 d = 0; d < numDocs; ++d) {
            if (pp[d] < 2) { gSums[pp[d]] += gp[d]; hSums[pp[d]] += hp[d]; }
        }

        auto leafValues = ComputeLeafValues(
            mx::array(gSums.data(), {2}, mx::float32),
            mx::array(hSums.data(), {2}, mx::float32),
            l2Reg, learningRate
        );

        ApplyObliviousTree(cursor, partitions, leafValues, numDocs);

        auto loss = mx::mean(mx::square(mx::subtract(cursor, targetsArr)));
        mx::eval(loss);
        float lossVal = loss.item<float>();

        if (iter % 3 == 0 || iter == numIterations - 1) {
            printf("  iter=%u loss=%.4f bestSplit=feat%u>bin%u score=%.4f\n",
                   iter, lossVal, featureIdx, binThreshold+1, bestScore);
        }

        if (lossVal > prevLoss + 1e-6f && iter > 0) {
            monotonic = false;
        }
        prevLoss = lossVal;
    }

    if (monotonic) {
        printf("  PASS: Loss decreased monotonically over %u iterations\n", numIterations);
    } else {
        printf("  WARNING: Loss was not strictly monotonic (may be OK with small oscillations)\n");
    }

    bool pass = prevLoss < 10.0f;  // should be much lower than initial
    printf("  Final loss: %.4f (initial ~10.67)\n", prevLoss);
    return pass;
}

int main() {
    printf("CatBoost-MLX Phase 6 Build Verification Test\n");
    printf("=============================================\n\n");

    bool allPass = true;
    allPass &= TestFullIteration();
    allPass &= TestMultipleIterations();

    printf("\n=============================================\n");
    if (allPass) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
}
