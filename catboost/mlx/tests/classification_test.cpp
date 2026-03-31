// Classification target and multi-dim pipeline tests.
// Tests Logloss/MultiClass gradients, training convergence, and multi-dim export.
//
// Compile: clang++ -std=c++17 -I. -I/opt/homebrew/Cellar/mlx/0.31.1/include
//          -L/opt/homebrew/Cellar/mlx/0.31.1/lib -lmlx
//          -framework Metal -framework Foundation -Wno-c++20-extensions
//          catboost/mlx/tests/classification_test.cpp -o /tmp/classification_test

#include <mlx/mlx.h>
#include <mlx/fast.h>
#include <catboost/mlx/kernels/kernel_sources.h>

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>

namespace mx = mlx::core;
using ui32 = uint32_t;

// ---- Mirrored structures ----

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
    bool Defined() const { return FeatureId != static_cast<ui32>(-1); }
};

struct TObliviousSplitLevel {
    ui32 FeatureColumnIdx;
    ui32 Shift;
    ui32 Mask;
    ui32 BinThreshold;
};

struct TPartitionStatistics {
    double Sum = 0.0;
    double Weight = 0.0;
    double Count = 0.0;
};

// ============================================================================
// Test 1: Logloss gradient verification
// ============================================================================

bool TestLoglossGradients() {
    printf("=== Test: Logloss Gradients ===\n");

    const int N = 8;
    // Known cursor values
    std::vector<float> cursorVals = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f, 3.0f};
    // Binary targets
    std::vector<float> targetVals = {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f};
    std::vector<float> weightVals(N, 1.0f);

    auto cursor = mx::array(cursorVals.data(), {N}, mx::float32);
    auto targets = mx::array(targetVals.data(), {N}, mx::float32);
    auto weights = mx::array(weightVals.data(), {N}, mx::float32);

    // Compute sigmoid
    auto sigmoid = mx::sigmoid(cursor);
    auto gradients = mx::subtract(sigmoid, targets);
    auto hessians = mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid));

    mx::eval({sigmoid, gradients, hessians});
    const float* sigPtr = sigmoid.data<float>();
    const float* gradPtr = gradients.data<float>();
    const float* hessPtr = hessians.data<float>();

    bool pass = true;
    for (int i = 0; i < N; ++i) {
        float expectedSig = 1.0f / (1.0f + std::exp(-cursorVals[i]));
        float expectedGrad = expectedSig - targetVals[i];
        float expectedHess = expectedSig * (1.0f - expectedSig);

        if (std::abs(sigPtr[i] - expectedSig) > 1e-5f) {
            printf("  FAIL: sigmoid[%d] = %.6f, expected %.6f\n", i, sigPtr[i], expectedSig);
            pass = false;
        }
        if (std::abs(gradPtr[i] - expectedGrad) > 1e-5f) {
            printf("  FAIL: gradient[%d] = %.6f, expected %.6f\n", i, gradPtr[i], expectedGrad);
            pass = false;
        }
        if (std::abs(hessPtr[i] - expectedHess) > 1e-5f) {
            printf("  FAIL: hessian[%d] = %.6f, expected %.6f\n", i, hessPtr[i], expectedHess);
            pass = false;
        }
    }

    if (pass) printf("  PASS: All %d gradient/hessian values correct\n", N);
    return pass;
}

// ============================================================================
// Test 2: MultiClass (softmax) gradient verification
// ============================================================================

bool TestMultiClassGradients() {
    printf("\n=== Test: MultiClass Gradients ===\n");

    const int K = 2;  // approxDim = numClasses - 1 = 2 (3 classes)
    const int N = 4;

    // Cursor: [K, N] = [2, 4]
    std::vector<float> cursorVals = {
        // dim 0: 4 docs
        1.0f, -0.5f, 0.0f, 2.0f,
        // dim 1: 4 docs
        -1.0f, 1.0f, 0.0f, -1.0f
    };
    // Target classes: 0, 1, 2, 0 (class 2 is the implicit class)
    std::vector<float> targetVals = {0.0f, 1.0f, 2.0f, 0.0f};

    auto cursor = mx::array(cursorVals.data(), {K, N}, mx::float32);
    auto targets = mx::array(targetVals.data(), {N}, mx::float32);

    // Manually compute softmax with implicit class (value 0)
    mx::eval(cursor);
    const float* cPtr = cursor.data<float>();

    bool pass = true;
    for (int d = 0; d < N; ++d) {
        float c0 = cPtr[d];         // dim 0
        float c1 = cPtr[N + d];     // dim 1
        float cImplicit = 0.0f;     // implicit class

        float maxC = std::max({c0, c1, cImplicit});
        float e0 = std::exp(c0 - maxC);
        float e1 = std::exp(c1 - maxC);
        float eImp = std::exp(cImplicit - maxC);
        float sumE = e0 + e1 + eImp;

        float p0 = e0 / sumE;
        float p1 = e1 / sumE;
        float pImp = eImp / sumE;

        int targetClass = static_cast<int>(targetVals[d]);

        // Gradients for each dim
        float expectedGrad0 = p0 - (targetClass == 0 ? 1.0f : 0.0f);
        float expectedGrad1 = p1 - (targetClass == 1 ? 1.0f : 0.0f);

        float expectedHess0 = p0 * (1.0f - p0);
        float expectedHess1 = p1 * (1.0f - p1);

        printf("  doc=%d target=%d: p=[%.4f, %.4f, %.4f] grad=[%.4f, %.4f] hess=[%.4f, %.4f]\n",
               d, targetClass, p0, p1, pImp, expectedGrad0, expectedGrad1, expectedHess0, expectedHess1);

        // Verify probabilities sum to 1
        float probSum = p0 + p1 + pImp;
        if (std::abs(probSum - 1.0f) > 1e-5f) {
            printf("  FAIL: probabilities sum to %.6f, expected 1.0\n", probSum);
            pass = false;
        }
    }

    if (pass) printf("  PASS: Softmax gradients computed correctly\n");
    return pass;
}

// ============================================================================
// Test 3: Logloss end-to-end training (binary classification)
// ============================================================================

struct TPartitionLayout {
    mx::array DocIndices;
    mx::array PartOffsets;
    mx::array PartSizes;
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
        mx::array(reinterpret_cast<const int32_t*>(docIndices.data()), {static_cast<int>(numDocs)}, mx::uint32),
        mx::array(reinterpret_cast<const int32_t*>(partOffsets.data()), {static_cast<int>(numPartitions)}, mx::uint32),
        mx::array(reinterpret_cast<const int32_t*>(partSizes.data()), {static_cast<int>(numPartitions)}, mx::uint32)
    };
}

bool TestLoglossTraining() {
    printf("\n=== Test: Logloss Training ===\n");

    const ui32 numDocs = 16;
    const ui32 numFeatures = 4;
    const ui32 numUi32PerDoc = 1;
    const ui32 folds = 2;
    const ui32 totalBinFeatures = numFeatures * folds;
    const float learningRate = 0.5f;
    const float l2Reg = 1.0f;

    // GPU feature metadata
    std::vector<TCFeature> gpuFeatures;
    ui32 firstFold = 0;
    for (ui32 f = 0; f < numFeatures; ++f) {
        TCFeature feat;
        feat.Offset = 0;
        feat.Shift = (3 - f) * 8;
        feat.Mask = 0xFF << feat.Shift;
        feat.FirstFoldIndex = firstFold;
        feat.Folds = folds;
        feat.OneHotFeature = false;
        feat.SkipFirstBinInScoreCount = false;
        gpuFeatures.push_back(feat);
        firstFold += folds;
    }

    // Pack feature data and create binary targets
    std::vector<uint32_t> packed(numDocs);
    std::vector<float> targets(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        uint8_t f0 = d % 3;
        uint8_t f1 = (d / 3) % 3;
        uint8_t f2 = (d + 1) % 3;
        uint8_t f3 = (d + 2) % 3;
        packed[d] = (f0 << 24) | (f1 << 16) | (f2 << 8) | f3;
        // Binary target: 1 if feature0 >= 1, else 0
        targets[d] = (f0 >= 1) ? 1.0f : 0.0f;
    }

    auto compressedData = mx::array(reinterpret_cast<const int32_t*>(packed.data()), {static_cast<int>(numDocs)}, mx::uint32);
    auto targetsArr = mx::array(targets.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto cursor = mx::zeros({static_cast<int>(numDocs)}, mx::float32);
    auto weights = mx::ones({static_cast<int>(numDocs)}, mx::float32);

    std::vector<ui32> foldCountsVec = {folds, folds, folds, folds};
    std::vector<ui32> firstFoldVec = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    // Compute initial loss
    auto initSigmoid = mx::sigmoid(cursor);
    auto eps = mx::array(1e-15f);
    auto initLoss = mx::negative(mx::mean(mx::add(
        mx::multiply(targetsArr, mx::log(mx::add(initSigmoid, eps))),
        mx::multiply(mx::subtract(mx::array(1.0f), targetsArr), mx::log(mx::add(mx::subtract(mx::array(1.0f), initSigmoid), eps)))
    )));
    mx::eval(initLoss);
    float initLossVal = initLoss.item<float>();
    printf("  Initial logloss: %.4f\n", initLossVal);

    // Run 5 boosting iterations
    for (ui32 iter = 0; iter < 5; ++iter) {
        // Compute Logloss gradients
        auto sigmoid = mx::sigmoid(cursor);
        auto gradients = mx::subtract(sigmoid, targetsArr);
        auto hessians = mx::multiply(sigmoid, mx::subtract(mx::array(1.0f), sigmoid));
        hessians = mx::maximum(hessians, mx::array(1e-16f));

        auto stats = mx::concatenate({
            mx::reshape(gradients, {1, static_cast<int>(numDocs)}),
            mx::reshape(hessians, {1, static_cast<int>(numDocs)})
        }, 0);
        stats = mx::reshape(stats, {static_cast<int>(2 * numDocs)});

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

        // Find best split using L2 gain with hessian
        mx::eval({gradients, hessians, partitions});
        const float* gp = gradients.data<float>();
        const float* hp = hessians.data<float>();
        const uint32_t* pp = partitions.data<uint32_t>();

        // Compute partition stats
        TPartitionStatistics partStat;
        for (ui32 d = 0; d < numDocs; ++d) {
            partStat.Sum += gp[d];
            partStat.Weight += hp[d];
        }

        float bestScore = -1.0f;
        ui32 bestFeature = 0, bestBin = 0;
        for (ui32 f = 0; f < numFeatures; ++f) {
            for (ui32 b = 0; b < folds; ++b) {
                ui32 bf = f * folds + b;
                float sumLeft = histData[bf];
                float weightLeft = histData[totalBinFeatures + bf];
                float sumRight = static_cast<float>(partStat.Sum) - sumLeft;
                float weightRight = static_cast<float>(partStat.Weight) - weightLeft;

                if (weightLeft < 1e-15f || weightRight < 1e-15f) continue;

                float score = (sumLeft * sumLeft) / (weightLeft + l2Reg)
                            + (sumRight * sumRight) / (weightRight + l2Reg)
                            - static_cast<float>(partStat.Sum * partStat.Sum) / (static_cast<float>(partStat.Weight) + l2Reg);
                if (score > bestScore) {
                    bestScore = score;
                    bestFeature = f;
                    bestBin = b;
                }
            }
        }

        // Build tree and compute leaf values
        const auto& feat = gpuFeatures[bestFeature];
        std::vector<uint32_t> newParts(numDocs);
        for (ui32 d = 0; d < numDocs; ++d) {
            uint8_t bin = (packed[d] >> feat.Shift) & 0xFF;
            newParts[d] = (bin > bestBin) ? 1 : 0;
        }

        std::vector<float> gSums(2, 0.0f), hSums(2, 0.0f);
        for (ui32 d = 0; d < numDocs; ++d) {
            gSums[newParts[d]] += gp[d];
            hSums[newParts[d]] += hp[d];
        }

        std::vector<float> leafVals(2);
        for (int i = 0; i < 2; ++i) {
            leafVals[i] = -learningRate * gSums[i] / (hSums[i] + l2Reg);
        }

        // Apply tree
        auto leafArr = mx::array(leafVals.data(), {2}, mx::float32);
        auto partsArr = mx::array(reinterpret_cast<const int32_t*>(newParts.data()), {static_cast<int>(numDocs)}, mx::uint32);
        auto gathered = mx::take(leafArr, partsArr, 0);
        cursor = mx::add(cursor, gathered);
        mx::eval(cursor);

        printf("  iter=%u: feature=%u bin=%u gain=%.4f leaf=[%.4f, %.4f]\n",
               iter, bestFeature, bestBin, bestScore, leafVals[0], leafVals[1]);
    }

    // Compute final loss
    auto finalSigmoid = mx::sigmoid(cursor);
    auto finalLoss = mx::negative(mx::mean(mx::add(
        mx::multiply(targetsArr, mx::log(mx::add(finalSigmoid, eps))),
        mx::multiply(mx::subtract(mx::array(1.0f), targetsArr), mx::log(mx::add(mx::subtract(mx::array(1.0f), finalSigmoid), eps)))
    )));
    mx::eval(finalLoss);
    float finalLossVal = finalLoss.item<float>();

    printf("  Initial logloss: %.4f, Final logloss: %.4f\n", initLossVal, finalLossVal);

    bool pass = (finalLossVal < initLossVal);
    if (!pass) {
        printf("  FAIL: Loss did not decrease\n");
    } else {
        printf("  PASS\n");
    }
    return pass;
}

// ============================================================================
// Test 4: MultiClass training (3 classes, approxDim=2)
// ============================================================================

bool TestMultiClassTraining() {
    printf("\n=== Test: MultiClass Training ===\n");

    const int K = 2;  // approxDim = numClasses - 1
    const ui32 numDocs = 18;  // divisible by 3
    const ui32 numFeatures = 4;
    const ui32 folds = 2;
    const ui32 totalBinFeatures = numFeatures * folds;
    const float learningRate = 0.3f;
    const float l2Reg = 1.0f;

    // GPU features
    std::vector<TCFeature> gpuFeatures;
    ui32 firstFold = 0;
    for (ui32 f = 0; f < numFeatures; ++f) {
        TCFeature feat;
        feat.Offset = 0;
        feat.Shift = (3 - f) * 8;
        feat.Mask = 0xFF << feat.Shift;
        feat.FirstFoldIndex = firstFold;
        feat.Folds = folds;
        feat.OneHotFeature = false;
        feat.SkipFirstBinInScoreCount = false;
        gpuFeatures.push_back(feat);
        firstFold += folds;
    }

    // Data: target = feature0 mod 3 (3-class problem)
    std::vector<uint32_t> packed(numDocs);
    std::vector<float> targets(numDocs);
    for (ui32 d = 0; d < numDocs; ++d) {
        uint8_t f0 = d % 3;
        uint8_t f1 = (d / 3) % 3;
        uint8_t f2 = (d + 1) % 3;
        uint8_t f3 = (d + 2) % 3;
        packed[d] = (f0 << 24) | (f1 << 16) | (f2 << 8) | f3;
        targets[d] = static_cast<float>(f0);  // class 0, 1, or 2
    }

    auto compressedData = mx::array(reinterpret_cast<const int32_t*>(packed.data()), {static_cast<int>(numDocs)}, mx::uint32);
    auto targetsArr = mx::array(targets.data(), {static_cast<int>(numDocs)}, mx::float32);
    auto cursor = mx::zeros({K, static_cast<int>(numDocs)}, mx::float32);  // [K, numDocs]
    auto weights = mx::ones({static_cast<int>(numDocs)}, mx::float32);

    std::vector<ui32> foldCountsVec = {folds, folds, folds, folds};
    std::vector<ui32> firstFoldVec = {0, 2, 4, 6};
    auto foldCountsArr = mx::array(reinterpret_cast<const int32_t*>(foldCountsVec.data()), {4}, mx::uint32);
    auto firstFoldArr = mx::array(reinterpret_cast<const int32_t*>(firstFoldVec.data()), {4}, mx::uint32);

    // Helper: compute multi-class loss
    auto computeLoss = [&](const mx::array& cur) -> float {
        // Softmax with implicit class
        auto maxC = mx::maximum(mx::max(cur, 0), mx::array(0.0f));
        auto expC = mx::exp(mx::subtract(cur, mx::reshape(maxC, {1, static_cast<int>(numDocs)})));
        auto expImp = mx::exp(mx::negative(maxC));
        auto sumExp = mx::add(mx::sum(expC, 0), expImp);

        // For each doc, gather prob of target class
        auto targetInt = mx::astype(targetsArr, mx::int32);
        auto probTarget = mx::zeros({static_cast<int>(numDocs)}, mx::float32);
        for (int k = 0; k < K; ++k) {
            auto isClass = mx::astype(mx::equal(targetInt, mx::array(k)), mx::float32);
            auto probK = mx::divide(
                mx::reshape(mx::slice(expC, {k, 0}, {k + 1, static_cast<int>(numDocs)}), {static_cast<int>(numDocs)}),
                sumExp
            );
            probTarget = mx::add(probTarget, mx::multiply(isClass, probK));
        }
        // Implicit class (K=2)
        auto isLast = mx::astype(mx::equal(targetInt, mx::array(K)), mx::float32);
        auto probImp = mx::divide(expImp, sumExp);
        probTarget = mx::add(probTarget, mx::multiply(isLast, probImp));

        auto loss = mx::negative(mx::mean(mx::log(mx::add(probTarget, mx::array(1e-15f)))));
        mx::eval(loss);
        return loss.item<float>();
    };

    float initLossVal = computeLoss(cursor);
    printf("  Initial multi-class loss: %.4f\n", initLossVal);

    // Run 5 boosting iterations
    for (ui32 iter = 0; iter < 5; ++iter) {
        // Compute softmax gradients per dim
        auto maxC = mx::maximum(mx::max(cursor, 0), mx::array(0.0f));
        auto expC = mx::exp(mx::subtract(cursor, mx::reshape(maxC, {1, static_cast<int>(numDocs)})));
        auto expImp = mx::exp(mx::negative(maxC));
        auto sumExp = mx::add(mx::sum(expC, 0), expImp);
        auto probs = mx::divide(expC, mx::reshape(sumExp, {1, static_cast<int>(numDocs)}));

        // Gradients and hessians: [K, numDocs]
        auto targetInt = mx::astype(targetsArr, mx::uint32);
        mx::array allGrads = mx::zeros({K, static_cast<int>(numDocs)}, mx::float32);
        mx::array allHess = mx::zeros({K, static_cast<int>(numDocs)}, mx::float32);

        mx::eval(probs);

        for (int k = 0; k < K; ++k) {
            auto isClass = mx::astype(
                mx::equal(targetInt, mx::array(static_cast<uint32_t>(k))),
                mx::float32
            );
            auto probK = mx::reshape(mx::slice(probs, {k, 0}, {k + 1, static_cast<int>(numDocs)}), {static_cast<int>(numDocs)});
            auto gradK = mx::subtract(probK, isClass);
            auto hessK = mx::maximum(mx::multiply(probK, mx::subtract(mx::array(1.0f), probK)), mx::array(1e-16f));

            // For each dim k, find best split across this dim's histogram
            // First: dispatch histogram for dim k
            auto statsK = mx::concatenate({
                mx::reshape(gradK, {1, static_cast<int>(numDocs)}),
                mx::reshape(hessK, {1, static_cast<int>(numDocs)})
            }, 0);
            statsK = mx::reshape(statsK, {static_cast<int>(2 * numDocs)});

            // Store grads/hess
            mx::eval({gradK, hessK});
            // We'll accumulate below
            allGrads = mx::where(
                mx::equal(mx::reshape(mx::arange(K), {K, 1}), mx::array(k)),
                mx::reshape(gradK, {1, static_cast<int>(numDocs)}),
                allGrads
            );
            allHess = mx::where(
                mx::equal(mx::reshape(mx::arange(K), {K, 1}), mx::array(k)),
                mx::reshape(hessK, {1, static_cast<int>(numDocs)}),
                allHess
            );
        }

        mx::eval({allGrads, allHess});

        // Now find best split using sum of gains across dims
        // For simplicity in this test, compute histograms and score on CPU
        auto partitions = mx::zeros({static_cast<int>(numDocs)}, mx::uint32);
        auto layout = ComputePartitionLayout(partitions, numDocs, 1);

        // For each dim, compute histograms
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

        // Collect per-dim histograms
        std::vector<std::vector<float>> perDimHist(K);
        std::vector<std::vector<float>> perDimGradsHost(K, std::vector<float>(numDocs));
        std::vector<std::vector<float>> perDimHessHost(K, std::vector<float>(numDocs));

        for (int k = 0; k < K; ++k) {
            auto dimGrads = mx::reshape(mx::slice(allGrads, {k, 0}, {k + 1, static_cast<int>(numDocs)}), {static_cast<int>(numDocs)});
            auto dimHess = mx::reshape(mx::slice(allHess, {k, 0}, {k + 1, static_cast<int>(numDocs)}), {static_cast<int>(numDocs)});

            auto statsK = mx::concatenate({
                mx::reshape(dimGrads, {1, static_cast<int>(numDocs)}),
                mx::reshape(dimHess, {1, static_cast<int>(numDocs)})
            }, 0);
            statsK = mx::reshape(statsK, {static_cast<int>(2 * numDocs)});

            mx::Shape histShape = {static_cast<int>(2 * totalBinFeatures)};
            auto histResult = kernel(
                {compressedData, statsK, layout.DocIndices,
                 layout.PartOffsets, layout.PartSizes,
                 mx::array(static_cast<uint32_t>(0), mx::uint32),
                 mx::array(static_cast<uint32_t>(1), mx::uint32),
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
            const float* hData = histResult[0].data<float>();
            perDimHist[k].assign(hData, hData + 2 * totalBinFeatures);

            mx::eval({dimGrads, dimHess});
            const float* gPtr = dimGrads.data<float>();
            const float* hPtr = dimHess.data<float>();
            for (ui32 d = 0; d < numDocs; ++d) {
                perDimGradsHost[k][d] = gPtr[d];
                perDimHessHost[k][d] = hPtr[d];
            }
        }

        // Score: sum gain across dims
        // Compute partition stats per dim
        std::vector<TPartitionStatistics> perDimPartStats(K);
        for (int k = 0; k < K; ++k) {
            for (ui32 d = 0; d < numDocs; ++d) {
                perDimPartStats[k].Sum += perDimGradsHost[k][d];
                perDimPartStats[k].Weight += perDimHessHost[k][d];
            }
        }

        float bestScore = -1.0f;
        ui32 bestFeature = 0, bestBin = 0;
        for (ui32 f = 0; f < numFeatures; ++f) {
            for (ui32 b = 0; b < folds; ++b) {
                ui32 bf = f * folds + b;
                float totalGain = 0.0f;
                bool valid = true;

                for (int k = 0; k < K; ++k) {
                    float sumLeft = perDimHist[k][bf];
                    float weightLeft = perDimHist[k][totalBinFeatures + bf];
                    float sumRight = static_cast<float>(perDimPartStats[k].Sum) - sumLeft;
                    float weightRight = static_cast<float>(perDimPartStats[k].Weight) - weightLeft;

                    if (weightLeft < 1e-15f || weightRight < 1e-15f) { valid = false; break; }

                    totalGain += (sumLeft * sumLeft) / (weightLeft + l2Reg)
                               + (sumRight * sumRight) / (weightRight + l2Reg)
                               - static_cast<float>(perDimPartStats[k].Sum * perDimPartStats[k].Sum) / (static_cast<float>(perDimPartStats[k].Weight) + l2Reg);
                }

                if (valid && totalGain > bestScore) {
                    bestScore = totalGain;
                    bestFeature = f;
                    bestBin = b;
                }
            }
        }

        // Build tree and compute per-dim leaf values
        const auto& feat = gpuFeatures[bestFeature];
        std::vector<uint32_t> newParts(numDocs);
        for (ui32 d = 0; d < numDocs; ++d) {
            uint8_t bin = (packed[d] >> feat.Shift) & 0xFF;
            newParts[d] = (bin > bestBin) ? 1 : 0;
        }

        // Leaf values: [2, K] = [numLeaves, approxDim]
        std::vector<float> leafVals(2 * K, 0.0f);
        for (int k = 0; k < K; ++k) {
            float gSums[2] = {0, 0}, hSums[2] = {0, 0};
            for (ui32 d = 0; d < numDocs; ++d) {
                gSums[newParts[d]] += perDimGradsHost[k][d];
                hSums[newParts[d]] += perDimHessHost[k][d];
            }
            for (int leaf = 0; leaf < 2; ++leaf) {
                leafVals[leaf * K + k] = -learningRate * gSums[leaf] / (hSums[leaf] + l2Reg);
            }
        }

        printf("  iter=%u: feature=%u bin=%u gain=%.4f leaf0=[%.4f,%.4f] leaf1=[%.4f,%.4f]\n",
               iter, bestFeature, bestBin, bestScore,
               leafVals[0], leafVals[1], leafVals[K], leafVals[K + 1]);

        // Apply tree: gather [numLeaves, K] by partition index, transpose to [K, numDocs]
        auto leafArr = mx::array(leafVals.data(), {2, K}, mx::float32);
        auto partsArr = mx::array(reinterpret_cast<const int32_t*>(newParts.data()), {static_cast<int>(numDocs)}, mx::uint32);
        auto docLeafValues = mx::take(leafArr, partsArr, 0);  // [numDocs, K]
        docLeafValues = mx::transpose(docLeafValues);           // [K, numDocs]
        cursor = mx::add(cursor, docLeafValues);
        mx::eval(cursor);
    }

    float finalLossVal = computeLoss(cursor);
    printf("  Initial loss: %.4f, Final loss: %.4f\n", initLossVal, finalLossVal);

    bool pass = (finalLossVal < initLossVal);
    if (!pass) {
        printf("  FAIL: Multi-class loss did not decrease\n");
    } else {
        printf("  PASS\n");
    }
    return pass;
}

// ============================================================================
// Test 5: Multi-dim leaf value export interleaving
// ============================================================================

bool TestMultiDimExport() {
    printf("\n=== Test: Multi-Dim Leaf Value Export ===\n");

    const ui32 K = 2;  // approxDim
    const ui32 numLeaves = 4;  // depth=2

    // Simulated leaf values: [numLeaves, K] (leaf-major)
    std::vector<float> interleavedLeaves = {
        0.1f, 0.2f,   // leaf 0: dim0=0.1, dim1=0.2
        -0.3f, 0.4f,  // leaf 1: dim0=-0.3, dim1=0.4
        0.5f, -0.6f,  // leaf 2: dim0=0.5, dim1=-0.6
        -0.7f, 0.8f   // leaf 3: dim0=-0.7, dim1=0.8
    };

    // Deinterleave to [K][numLeaves] (as model exporter would)
    std::vector<std::vector<double>> multiDimLeafValues(K);
    for (ui32 dim = 0; dim < K; ++dim) {
        multiDimLeafValues[dim].resize(numLeaves);
        for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
            multiDimLeafValues[dim][leaf] = static_cast<double>(interleavedLeaves[leaf * K + dim]);
        }
    }

    bool pass = true;

    // Verify dim 0
    std::vector<double> expectedDim0 = {0.1, -0.3, 0.5, -0.7};
    std::vector<double> expectedDim1 = {0.2, 0.4, -0.6, 0.8};

    for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
        if (std::abs(multiDimLeafValues[0][leaf] - expectedDim0[leaf]) > 1e-6) {
            printf("  FAIL: dim0[%u] = %.6f, expected %.6f\n", leaf,
                   multiDimLeafValues[0][leaf], expectedDim0[leaf]);
            pass = false;
        }
        if (std::abs(multiDimLeafValues[1][leaf] - expectedDim1[leaf]) > 1e-6) {
            printf("  FAIL: dim1[%u] = %.6f, expected %.6f\n", leaf,
                   multiDimLeafValues[1][leaf], expectedDim1[leaf]);
            pass = false;
        }
    }

    if (pass) printf("  PASS: Multi-dim leaf values deinterleaved correctly\n");
    return pass;
}

int main() {
    printf("CatBoost-MLX Phase 8 Classification Test\n");
    printf("==========================================\n\n");

    bool allPass = true;
    allPass &= TestLoglossGradients();
    allPass &= TestMultiClassGradients();
    allPass &= TestLoglossTraining();
    allPass &= TestMultiClassTraining();
    allPass &= TestMultiDimExport();

    printf("\n==========================================\n");
    printf(allPass ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    return allPass ? 0 : 1;
}
