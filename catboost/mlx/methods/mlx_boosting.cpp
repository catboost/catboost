#include "mlx_boosting.h"
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/libs/logging/logging.h>

#include <chrono>

namespace NCatboostMlx {

    TBoostingResult RunBoosting(
        TMLXDataSet& trainData,
        const IMLXTargetFunc& target,
        const TBoostingConfig& config,
        ITrainingCallbacks* callbacks,
        TMetricsAndTimeLeftHistory* metricsHistory
    ) {
        TBoostingResult result;
        result.TreeStructures.reserve(config.NumIterations);
        result.TreeLeafValues.reserve(config.NumIterations);
        result.ApproxDimension = config.ApproxDimension;

        const ui32 numDocs = trainData.GetNumDocs();
        const ui32 approxDim = config.ApproxDimension;

        CATBOOST_INFO_LOG << "CatBoost-MLX: Starting boosting with "
            << config.NumIterations << " iterations, lr=" << config.LearningRate
            << ", depth=" << config.MaxDepth << ", l2=" << config.L2RegLambda
            << ", docs=" << numDocs << ", approxDim=" << approxDim << Endl;

        auto startTime = std::chrono::steady_clock::now();

        for (ui32 iter = 0; iter < config.NumIterations; ++iter) {
            auto iterStart = std::chrono::steady_clock::now();

            // ----- Step 1: Compute gradients and hessians -----
            mx::array cursor;
            if (approxDim == 1) {
                cursor = mx::reshape(trainData.GetCursor(), {static_cast<int>(numDocs)});
            } else {
                cursor = mx::reshape(trainData.GetCursor(),
                    {static_cast<int>(approxDim), static_cast<int>(numDocs)});
            }
            target.ComputeDerivatives(
                cursor,
                trainData.GetTargets(),
                trainData.GetWeights(),
                trainData.GetGradients(),
                trainData.GetHessians()
            );

            // ----- Step 2: Search for best tree structure -----
            trainData.InitPartitions(numDocs);

            auto treeStructure = SearchTreeStructure(
                trainData,
                config.MaxDepth,
                config.L2RegLambda,
                approxDim
            );

            if (treeStructure.Splits.empty()) {
                CATBOOST_WARNING_LOG << "CatBoost-MLX: No valid splits at iteration " << iter
                    << ", stopping early" << Endl;
                break;
            }

            // ----- Step 3: Estimate leaf values -----
            const ui32 numLeaves = 1u << treeStructure.Splits.size();

            TMLXDevice::EvalNow({trainData.GetGradients(), trainData.GetHessians(), trainData.GetPartitions()});
            const uint32_t* partsPtr = trainData.GetPartitions().data<uint32_t>();

            mx::array leafValues;

            if (approxDim == 1) {
                // Single-dim: accumulate scalar grad/hess sums per leaf
                auto flatGrads = mx::reshape(trainData.GetGradients(), {static_cast<int>(numDocs)});
                auto flatHess = mx::reshape(trainData.GetHessians(), {static_cast<int>(numDocs)});
                TMLXDevice::EvalNow({flatGrads, flatHess});

                const float* gradsPtr = flatGrads.data<float>();
                const float* hessPtr = flatHess.data<float>();

                TVector<float> gradSumsHost(numLeaves, 0.0f);
                TVector<float> hessSumsHost(numLeaves, 0.0f);
                for (ui32 d = 0; d < numDocs; ++d) {
                    ui32 leaf = partsPtr[d];
                    if (leaf < numLeaves) {
                        gradSumsHost[leaf] += gradsPtr[d];
                        hessSumsHost[leaf] += hessPtr[d];
                    }
                }

                auto gradSums = mx::array(gradSumsHost.data(), {static_cast<int>(numLeaves)}, mx::float32);
                auto hessSums = mx::array(hessSumsHost.data(), {static_cast<int>(numLeaves)}, mx::float32);

                leafValues = ComputeLeafValues(gradSums, hessSums, config.L2RegLambda, config.LearningRate);
                // leafValues shape: [numLeaves]
            } else {
                // Multi-dim: accumulate [K, numLeaves] grad/hess sums
                TVector<TVector<float>> gradSumsPerDim(approxDim, TVector<float>(numLeaves, 0.0f));
                TVector<TVector<float>> hessSumsPerDim(approxDim, TVector<float>(numLeaves, 0.0f));

                for (ui32 k = 0; k < approxDim; ++k) {
                    auto dimGrads = mx::slice(trainData.GetGradients(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimGrads = mx::reshape(dimGrads, {static_cast<int>(numDocs)});
                    auto dimHess = mx::slice(trainData.GetHessians(),
                        {static_cast<int>(k), 0}, {static_cast<int>(k + 1), static_cast<int>(numDocs)});
                    dimHess = mx::reshape(dimHess, {static_cast<int>(numDocs)});
                    TMLXDevice::EvalNow({dimGrads, dimHess});

                    const float* gPtr = dimGrads.data<float>();
                    const float* hPtr = dimHess.data<float>();

                    for (ui32 d = 0; d < numDocs; ++d) {
                        ui32 leaf = partsPtr[d];
                        if (leaf < numLeaves) {
                            gradSumsPerDim[k][leaf] += gPtr[d];
                            hessSumsPerDim[k][leaf] += hPtr[d];
                        }
                    }
                }

                // Compute leaf values per dimension, build [numLeaves, K]
                TVector<float> interleavedLeaves(numLeaves * approxDim, 0.0f);
                for (ui32 k = 0; k < approxDim; ++k) {
                    auto gradSums = mx::array(gradSumsPerDim[k].data(), {static_cast<int>(numLeaves)}, mx::float32);
                    auto hessSums = mx::array(hessSumsPerDim[k].data(), {static_cast<int>(numLeaves)}, mx::float32);
                    auto dimLeafValues = ComputeLeafValues(gradSums, hessSums, config.L2RegLambda, config.LearningRate);
                    TMLXDevice::EvalNow(dimLeafValues);
                    const float* lvPtr = dimLeafValues.data<float>();
                    for (ui32 leaf = 0; leaf < numLeaves; ++leaf) {
                        interleavedLeaves[leaf * approxDim + k] = lvPtr[leaf];
                    }
                }
                leafValues = mx::array(interleavedLeaves.data(),
                    {static_cast<int>(numLeaves), static_cast<int>(approxDim)}, mx::float32);
                // leafValues shape: [numLeaves, K]
            }

            // ----- Step 4: Apply tree to predictions -----
            ApplyObliviousTree(trainData, treeStructure.Splits, leafValues, approxDim);

            // ----- Step 5: Report metrics -----
            result.TreeStructures.push_back(std::move(treeStructure));
            result.TreeLeafValues.push_back(leafValues);

            auto iterEnd = std::chrono::steady_clock::now();
            auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

            if (iter % 100 == 0 || iter == config.NumIterations - 1) {
                mx::array lossCursor;
                if (approxDim == 1) {
                    lossCursor = mx::reshape(trainData.GetCursor(), {static_cast<int>(numDocs)});
                } else {
                    lossCursor = mx::reshape(trainData.GetCursor(),
                        {static_cast<int>(approxDim), static_cast<int>(numDocs)});
                }
                auto loss = target.ComputeLoss(
                    lossCursor,
                    trainData.GetTargets(),
                    trainData.GetWeights()
                );
                float lossVal = loss.item<float>();

                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    iterEnd - startTime).count();

                CATBOOST_INFO_LOG << "CatBoost-MLX: iter=" << iter
                    << " loss=" << lossVal
                    << " time=" << iterMs << "ms"
                    << " total=" << elapsed << "s" << Endl;
            }

            // Check for early stopping via callbacks
            if (callbacks && metricsHistory) {
                if (!callbacks->IsContinueTraining(*metricsHistory)) {
                    CATBOOST_INFO_LOG << "CatBoost-MLX: Early stopping at iteration " << iter << Endl;
                    break;
                }
            }
        }

        result.NumIterations = result.TreeStructures.size();

        auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        CATBOOST_INFO_LOG << "CatBoost-MLX: Training complete. "
            << result.NumIterations << " trees in " << totalTime << "s" << Endl;

        return result;
    }

}  // namespace NCatboostMlx
