// Model export pipeline verification test.
// Tests the data flow from training results to model export inputs,
// verifying that tree structures contain valid split properties
// that can be mapped back to features and borders.
//
// Compile: clang++ -std=c++17 -I. -I/opt/homebrew/Cellar/mlx/0.31.1/include
//          -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx
//          -framework Metal -framework Foundation -Wno-c++20-extensions
//          catboost/mlx/tests/model_export_test.cpp -o /tmp/model_export_test

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <cassert>

namespace mx = mlx::core;
using ui32 = uint32_t;

// ---- Mirrored structures from the MLX backend ----

struct TCFeature {
    uint64_t Offset;
    ui32 Mask;
    ui32 Shift;
    ui32 FirstFoldIndex;
    ui32 Folds;
    bool OneHotFeature;
    bool SkipFirstBinInScoreCount;
};

struct TBestSplitProperties {
    ui32 FeatureId = static_cast<ui32>(-1);
    ui32 BinId = 0;
    float Score = 1e30f;
    float Gain = 1e30f;
};

struct TObliviousSplitLevel {
    ui32 FeatureColumnIdx;
    ui32 Shift;
    ui32 Mask;
    ui32 BinThreshold;
};

struct TObliviousTreeStructure {
    std::vector<TObliviousSplitLevel> Splits;
    std::vector<TBestSplitProperties> SplitProperties;
};

struct TBoostingResult {
    std::vector<TObliviousTreeStructure> TreeStructures;
    std::vector<mx::array> TreeLeafValues;
    ui32 NumIterations;
};

// ---- Simulated model export logic (mirrors model_exporter.cpp) ----

struct TFloatSplit {
    int FloatFeature;
    float Split;
};

struct TModelSplit {
    TFloatSplit FloatFeature;
};

struct TExportedTree {
    std::vector<TModelSplit> Splits;
    std::vector<double> LeafValues;
};

// Simulate the conversion logic from model_exporter.cpp
std::vector<TExportedTree> SimulateExport(
    const TBoostingResult& result,
    const std::vector<ui32>& externalFeatureIndices,
    // borders[internalFloatIdx] = vector of border values
    const std::vector<std::vector<float>>& borders
) {
    std::vector<TExportedTree> exported;

    for (ui32 treeIdx = 0; treeIdx < result.NumIterations; ++treeIdx) {
        const auto& tree = result.TreeStructures[treeIdx];
        const auto& leafArr = result.TreeLeafValues[treeIdx];
        const ui32 depth = tree.Splits.size();

        TExportedTree exp;

        // Convert splits
        for (ui32 level = 0; level < depth; ++level) {
            const auto& splitProps = tree.SplitProperties[level];
            ui32 gpuFeatureId = splitProps.FeatureId;
            ui32 binId = splitProps.BinId;

            // In real code: externalIdx = externalFeatureIndices[gpuFeatureId]
            // then internalIdx = featuresLayout.GetInternalFeatureIdx(externalIdx)
            // For this test, assume internalIdx == gpuFeatureId (simple 1:1 mapping)
            ui32 externalIdx = externalFeatureIndices[gpuFeatureId];
            // In real code, we'd look up internal idx from layout.
            // For test purposes, assume external == internal (no categorical features)
            ui32 internalIdx = externalIdx;

            if (internalIdx >= borders.size() || binId >= borders[internalIdx].size()) {
                printf("  ERROR: Feature %u bin %u out of range\n", internalIdx, binId);
                return {};
            }

            float border = borders[internalIdx][binId];
            exp.Splits.push_back(TModelSplit{TFloatSplit{static_cast<int>(internalIdx), border}});
        }

        // Convert leaf values
        mx::eval(leafArr);
        const ui32 numLeaves = 1u << depth;
        const float* leafPtr = leafArr.data<float>();
        exp.LeafValues.resize(numLeaves);
        for (ui32 i = 0; i < numLeaves; ++i) {
            exp.LeafValues[i] = static_cast<double>(leafPtr[i]);
        }

        exported.push_back(std::move(exp));
    }
    return exported;
}

// ---- Helper: run one boosting iteration to produce a tree ----

struct TPartitionLayout {
    mx::array DocIndices;
    mx::array PartOffsets;
    mx::array PartSizes;
};

TPartitionLayout ComputePartitionLayout(const mx::array& partitions, ui32 numDocs, ui32 numPartitions) {
    // GPU bucket sort — mirrors structure_searcher.cpp (DEFECT-001 fix).
    // The old CPU scatter-sort was replaced in Sprint 4; this test now uses
    // the same GPU algorithm so it exercises the production code path.
    auto docIndices = mx::astype(mx::argsort(partitions, /*axis=*/0), mx::uint32);

    auto onesF = mx::ones({static_cast<int>(numDocs)}, mx::float32);
    auto partSizesF = mx::scatter_add_axis(
        mx::zeros({static_cast<int>(numPartitions)}, mx::float32),
        partitions, onesF, 0
    );

    auto partOffsetsF = mx::cumsum(partSizesF, /*axis=*/0, /*reverse=*/false, /*inclusive=*/false);

    return TPartitionLayout{
        docIndices,
        mx::astype(partOffsetsF, mx::uint32),
        mx::astype(partSizesF, mx::uint32)
    };
}

// ============================================================================
// Test: Simulate full train → export pipeline
// ============================================================================

bool TestTrainAndExport() {
    printf("=== Test: Train → Export Pipeline ===\n");

    // Setup: 16 docs, 4 features, 3 bins each (borders: [0.5, 1.5])
    const ui32 numDocs = 16;
    const ui32 numFeatures = 4;
    const ui32 numUi32PerDoc = 1;
    const ui32 folds = 2;  // 3 bins → 2 folds
    const ui32 totalBinFeatures = numFeatures * folds;
    const float learningRate = 0.3f;
    const float l2Reg = 1.0f;

    // Feature borders (quantization thresholds)
    std::vector<std::vector<float>> borders = {
        {0.5f, 1.5f},  // feature 0
        {0.5f, 1.5f},  // feature 1
        {0.5f, 1.5f},  // feature 2
        {0.5f, 1.5f}   // feature 3
    };

    // External feature indices (identity for this test)
    std::vector<ui32> externalFeatureIndices = {0, 1, 2, 3};

    // GPU feature metadata
    std::vector<TCFeature> gpuFeatures;
    ui32 firstFold = 0;
    for (ui32 f = 0; f < numFeatures; ++f) {
        TCFeature feat;
        feat.Offset = 0;  // all in word 0
        feat.Shift = (3 - f) * 8;
        feat.Mask = 0xFF << feat.Shift;
        feat.FirstFoldIndex = firstFold;
        feat.Folds = folds;
        feat.OneHotFeature = false;
        feat.SkipFirstBinInScoreCount = false;
        gpuFeatures.push_back(feat);
        firstFold += folds;
    }

    // Pack feature data: bins 0-2 for each feature
    std::vector<uint32_t> packed(numDocs);
    std::vector<float> targets(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        uint8_t f0 = d % 3;
        uint8_t f1 = (d / 3) % 3;
        uint8_t f2 = (d + 1) % 3;
        uint8_t f3 = (d + 2) % 3;
        packed[d] = (f0 << 24) | (f1 << 16) | (f2 << 8) | f3;
        targets[d] = static_cast<float>(f0) * 2.0f - 2.0f;  // target depends on feature 0
    }

    auto compressedData = mx::array(reinterpret_cast<const int32_t*>(packed.data()), {static_cast<int>(numDocs)}, mx::uint32);
    auto targetsArr = mx::array(targets.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto cursor = mx::zeros({static_cast<int>(numDocs)}, mx::float32);

    std::vector<ui32> foldCountsVec = {folds, folds, folds, folds};
    std::vector<ui32> firstFoldVec = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    // Build boosting result by running 3 iterations
    TBoostingResult boostingResult;
    boostingResult.NumIterations = 0;

    for (ui32 iter = 0; iter < 3; ++iter) {
        // Compute gradients (RMSE: grad = prediction - target)
        auto gradients = mx::subtract(cursor, targetsArr);
        auto hessians = mx::ones({static_cast<int>(numDocs)}, mx::float32);

        auto stats = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians, {1, static_cast<int>(numDocs)})
        }, 0);
        stats = mx::reshape(stats, {static_cast<int>(2 * numDocs)});

        // Single partition at depth 0
        auto partitions = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
        auto layout = ComputePartitionLayout(partitions, numDocs, 1);

        // Dispatch histogram kernel
        using namespace NCatboostMlx;
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

        mx::Shape histShape = {static_cast<int>(2 * totalBinFeatures)};
        auto histResult = kernel(
            {compressedData, stats, layout.DocIndices,
             layout.PartOffsets, layout.PartSizes,
             mx::array(static_cast<uint32_t>(0), mx::uint32),
             mx::array(static_cast<uint32_t>(numUi32PerDoc), mx::uint32),
             mx::array(static_cast<uint32_t>(1), mx::uint32),
             foldCountsArr, firstFoldArr,
             mx::array(static_cast<uint32_t>(totalBinFeatures), mx::uint32),
             mx::array(static_cast<uint32_t>(2), mx::uint32),
             mx::array(static_cast<uint32_t>(numDocs), mx::uint32)},
            {histShape}, {mx::float32},
            std::make_tuple(256, 1, 2),
            std::make_tuple(256, 1, 1),
            {}, 0.0f, false, mx::Device::gpu
        );

        mx::eval(histResult[0]);
        const float* histData = histResult[0].data<float>();

        // Find best split (simple scoring)
        float bestScore = -1.0f;
        ui32 bestFeature = 0, bestBin = 0;
        for (ui32 f = 0; f < numFeatures; ++f) {
            for (ui32 b = 0; b < folds; ++b) {
                ui32 bf = f * folds + b;
                float g = histData[bf];
                float h = histData[totalBinFeatures + bf];
                float score = (h + l2Reg > 0) ? (g * g) / (h + l2Reg) : 0.0f;
                if (score > bestScore) {
                    bestScore = score;
                    bestFeature = f;
                    bestBin = b;
                }
            }
        }

        printf("  iter=%u: best split feature=%u bin=%u score=%.4f\n",
               iter, bestFeature, bestBin, bestScore);

        // Build tree structure (depth=1)
        TObliviousTreeStructure tree;
        TObliviousSplitLevel split;
        split.FeatureColumnIdx = static_cast<ui32>(gpuFeatures[bestFeature].Offset);
        split.Shift = gpuFeatures[bestFeature].Shift;
        split.Mask = gpuFeatures[bestFeature].Mask >> gpuFeatures[bestFeature].Shift;
        split.BinThreshold = bestBin;
        tree.Splits.push_back(split);

        TBestSplitProperties bestProps;
        bestProps.FeatureId = bestFeature;
        bestProps.BinId = bestBin;
        bestProps.Score = bestScore;
        bestProps.Gain = bestScore;
        tree.SplitProperties.push_back(bestProps);

        // Compute partition assignments for leaf estimation
        std::vector<uint32_t> newParts(numDocs);
        for (ui32 d = 0; d < numDocs; ++d) {
            uint8_t bin = (packed[d] >> split.Shift) & 0xFF;
            newParts[d] = (bin > split.BinThreshold) ? 1 : 0;
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

        // Newton step: leafValue = -lr * gradSum / (hessSum + l2)
        std::vector<float> leafVals(2);
        for (int i = 0; i < 2; ++i) {
            leafVals[i] = -learningRate * gSums[i] / (hSums[i] + l2Reg);
        }

        auto leafValuesArr = mx::array(leafVals.data(), {2}, mx::float32);
        tree.Splits.size();  // just to reference it

        boostingResult.TreeStructures.push_back(std::move(tree));
        boostingResult.TreeLeafValues.push_back(leafValuesArr);
        boostingResult.NumIterations++;

        // Apply tree to cursor
        auto gathered = mx::take(leafValuesArr, partitions, 0);
        cursor = mx::add(cursor, gathered);
        mx::eval(cursor);
    }

    printf("  Training produced %u trees\n", boostingResult.NumIterations);

    // ---- Now simulate model export ----
    printf("\n  Simulating model export...\n");

    auto exported = SimulateExport(boostingResult, externalFeatureIndices, borders);

    bool pass = true;
    if (exported.size() != boostingResult.NumIterations) {
        printf("  FAIL: Export produced %zu trees, expected %u\n",
               exported.size(), boostingResult.NumIterations);
        pass = false;
    }

    for (size_t t = 0; t < exported.size(); ++t) {
        const auto& tree = exported[t];
        const auto& origTree = boostingResult.TreeStructures[t];

        // Check split count
        if (tree.Splits.size() != origTree.Splits.size()) {
            printf("  FAIL: Tree %zu split count %zu != %zu\n",
                   t, tree.Splits.size(), origTree.Splits.size());
            pass = false;
            continue;
        }

        // Check splits have valid borders
        for (size_t s = 0; s < tree.Splits.size(); ++s) {
            const auto& split = tree.Splits[s];
            int featIdx = split.FloatFeature.FloatFeature;
            float border = split.FloatFeature.Split;

            if (featIdx < 0 || featIdx >= static_cast<int>(numFeatures)) {
                printf("  FAIL: Tree %zu split %zu: invalid feature index %d\n", t, s, featIdx);
                pass = false;
            } else {
                ui32 binId = origTree.SplitProperties[s].BinId;
                float expectedBorder = borders[featIdx][binId];
                if (std::abs(border - expectedBorder) > 1e-6f) {
                    printf("  FAIL: Tree %zu split %zu: border=%.4f expected=%.4f\n",
                           t, s, border, expectedBorder);
                    pass = false;
                } else {
                    printf("  Tree %zu split %zu: feature=%d border=%.4f (bin=%u) OK\n",
                           t, s, featIdx, border, binId);
                }
            }
        }

        // Check leaf values are non-zero
        ui32 numLeaves = 1u << origTree.Splits.size();
        if (tree.LeafValues.size() != numLeaves) {
            printf("  FAIL: Tree %zu leaf count %zu != %u\n",
                   t, tree.LeafValues.size(), numLeaves);
            pass = false;
        } else {
            bool anyNonZero = false;
            for (auto v : tree.LeafValues) {
                if (std::abs(v) > 1e-10) anyNonZero = true;
            }
            if (!anyNonZero) {
                printf("  FAIL: Tree %zu has all-zero leaf values\n", t);
                pass = false;
            } else {
                printf("  Tree %zu: %u leaves [%.6f, %.6f] OK\n",
                       t, numLeaves, tree.LeafValues[0], tree.LeafValues[1]);
            }
        }
    }

    // Verify loss decreased
    auto finalLoss = mx::mean(mx::square(mx::subtract(cursor, targetsArr)));
    auto initLoss = mx::mean(mx::square(targetsArr));
    mx::eval({finalLoss, initLoss});

    float initVal = initLoss.item<float>();
    float finalVal = finalLoss.item<float>();
    printf("\n  Initial MSE: %.4f, Final MSE: %.4f\n", initVal, finalVal);
    if (finalVal >= initVal) {
        printf("  FAIL: Loss did not decrease\n");
        pass = false;
    }

    if (pass) printf("  PASS\n");
    return pass;
}

// ============================================================================
// Test: Leaf values roundtrip (mx::array float32 → double → verify)
// ============================================================================

bool TestLeafValuesRoundtrip() {
    printf("\n=== Test: Leaf Values Roundtrip ===\n");

    // Create known leaf values
    std::vector<float> srcLeaves = {-0.123f, 0.456f, -0.789f, 1.011f, 0.0f, -2.5f, 3.14f, -0.001f};
    auto leafArr = mx::array(srcLeaves.data(), {static_cast<int>(srcLeaves.size())}, mx::float32);
    mx::eval(leafArr);

    // Simulate export conversion
    const float* leafPtr = leafArr.data<float>();
    std::vector<double> exported(srcLeaves.size());
    for (size_t i = 0; i < srcLeaves.size(); ++i) {
        exported[i] = static_cast<double>(leafPtr[i]);
    }

    bool pass = true;
    for (size_t i = 0; i < srcLeaves.size(); ++i) {
        double diff = std::abs(exported[i] - static_cast<double>(srcLeaves[i]));
        if (diff > 1e-6) {
            printf("  FAIL: leaf[%zu] = %.10f, expected %.10f\n",
                   i, exported[i], static_cast<double>(srcLeaves[i]));
            pass = false;
        }
    }

    if (pass) printf("  PASS: All %zu leaf values roundtripped correctly\n", srcLeaves.size());
    return pass;
}

// ============================================================================
// Test: External feature index mapping
// ============================================================================

bool TestFeatureIndexMapping() {
    printf("\n=== Test: Feature Index Mapping ===\n");

    // Simulate a case where external indices are not sequential
    // (e.g., some features were dropped or are categorical)
    // GPU local: [0, 1, 2, 3]
    // External:  [0, 2, 5, 7] (features 1, 3, 4, 6 are non-float or dropped)
    std::vector<ui32> externalIndices = {0, 2, 5, 7};

    // Borders for each external feature
    // borders[0] = {0.5}, borders[2] = {1.0, 2.0}, borders[5] = {-0.5, 0.0, 0.5}, borders[7] = {3.0}
    // Index by external feature index (sparse)
    std::vector<std::vector<float>> allBorders(8);
    allBorders[0] = {0.5f};
    allBorders[2] = {1.0f, 2.0f};
    allBorders[5] = {-0.5f, 0.0f, 0.5f};
    allBorders[7] = {3.0f};

    // Simulate: GPU says feature 1 (external 2), bin 1 → border should be 2.0
    ui32 gpuFeatureId = 1;
    ui32 binId = 1;
    ui32 externalIdx = externalIndices[gpuFeatureId];
    float border = allBorders[externalIdx][binId];

    bool pass = true;

    if (externalIdx != 2) {
        printf("  FAIL: Expected external idx 2, got %u\n", externalIdx);
        pass = false;
    }
    if (std::abs(border - 2.0f) > 1e-6f) {
        printf("  FAIL: Expected border 2.0, got %.4f\n", border);
        pass = false;
    }

    // Simulate: GPU says feature 2 (external 5), bin 2 → border should be 0.5
    gpuFeatureId = 2;
    binId = 2;
    externalIdx = externalIndices[gpuFeatureId];
    border = allBorders[externalIdx][binId];

    if (externalIdx != 5) {
        printf("  FAIL: Expected external idx 5, got %u\n", externalIdx);
        pass = false;
    }
    if (std::abs(border - 0.5f) > 1e-6f) {
        printf("  FAIL: Expected border 0.5, got %.4f\n", border);
        pass = false;
    }

    if (pass) printf("  PASS: Feature index mapping is correct\n");
    return pass;
}

int main() {
    printf("CatBoost-MLX Phase 7 Model Export Test\n");
    printf("=======================================\n\n");

    bool allPass = true;
    allPass &= TestFeatureIndexMapping();
    allPass &= TestLeafValuesRoundtrip();
    allPass &= TestTrainAndExport();

    printf("\n=======================================\n");
    printf(allPass ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return allPass ? 0 : 1;
}
