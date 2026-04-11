#include "mlx_boosting.h"
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/libs/logging/logging.h>

#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>

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

        // Validation state
        bool hasValidation = (config.ValidationData != nullptr);
        mx::array valCursor;
        float bestValLoss = std::numeric_limits<float>::infinity();
        ui32 bestIteration = 0;
        ui32 noImprovementCount = 0;

        if (hasValidation) {
            ui32 valDocs = config.ValidationData->GetNumDocs();
            valCursor = mx::zeros(
                {static_cast<int>(approxDim), static_cast<int>(valDocs)}, mx::float32);
            TMLXDevice::EvalNow(valCursor);
            CATBOOST_INFO_LOG << "CatBoost-MLX: Validation set: " << valDocs << " docs" << Endl;
        }

        // Random number generator for subsampling
        std::mt19937 rng(config.RandomSeed);

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

            // ----- Step 3: Estimate leaf values (GPU-accelerated) -----
            const ui32 numLeaves = 1u << treeStructure.Splits.size();

            mx::array gradSumsGPU, hessSumsGPU;
            ComputeLeafSumsGPU(
                trainData.GetGradients(),
                trainData.GetHessians(),
                trainData.GetPartitions(),
                numDocs, numLeaves, approxDim,
                gradSumsGPU, hessSumsGPU
            );

            mx::array leafValues;

            if (approxDim == 1) {
                // Scalar path: gradSumsGPU / hessSumsGPU are [numLeaves].
                // Returns a lazy array; materialised by ApplyObliviousTree or the
                // result store below.
                leafValues = ComputeLeafValues(gradSumsGPU, hessSumsGPU,
                    config.L2RegLambda, config.LearningRate);
                // leafValues shape: [numLeaves]
            } else {
                // Fused multiclass path — eliminates K EvalNow CPU-GPU round trips.
                //
                // gradSumsGPU / hessSumsGPU are [approxDim * numLeaves] laid out as:
                //   [dim0_leaf0, dim0_leaf1, ..., dim0_leafN, dim1_leaf0, ...]
                //   i.e. row-major [approxDim, numLeaves]
                //
                // ComputeLeafValues is element-wise so it applies identically over
                // the entire flat [approxDim * numLeaves] array in a single dispatch.
                // The result is then reshaped to [approxDim, numLeaves] and transposed
                // to [numLeaves, approxDim] — the interleaved layout that
                // ApplyObliviousTree / kTreeApplySource expect:
                //   leafValues[leafIdx * approxDim + k]
                auto flatLeafValues = ComputeLeafValues(gradSumsGPU, hessSumsGPU,
                    config.L2RegLambda, config.LearningRate);
                // flatLeafValues shape: [approxDim * numLeaves]
                auto dimByLeaf = mx::reshape(flatLeafValues,
                    {static_cast<int>(approxDim), static_cast<int>(numLeaves)});
                // dimByLeaf shape: [approxDim, numLeaves]
                leafValues = mx::transpose(dimByLeaf);
                // leafValues shape: [numLeaves, approxDim]
                // No EvalNow here — let the GPU graph continue lazily until
                // ApplyObliviousTree materialises it alongside the cursor update.
            }

            // ----- Step 4: Apply tree to predictions -----
            ApplyObliviousTree(trainData, treeStructure.Splits, leafValues, approxDim);

            // Apply tree to validation data if present
            if (hasValidation) {
                ApplyObliviousTree(*config.ValidationData, treeStructure.Splits, leafValues, approxDim);
            }

            // ----- Step 5: Report metrics -----
            result.TreeStructures.push_back(std::move(treeStructure));
            result.TreeLeafValues.push_back(leafValues);

            auto iterEnd = std::chrono::steady_clock::now();
            auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

            ui32 metricPeriod = std::max(config.MetricPeriod, 1u);
            if (iter % metricPeriod == 0 || iter == config.NumIterations - 1) {
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

                if (hasValidation) {
                    ui32 valDocs = config.ValidationData->GetNumDocs();
                    mx::array valLossCursor;
                    if (approxDim == 1) {
                        valLossCursor = mx::reshape(config.ValidationData->GetCursor(), {static_cast<int>(valDocs)});
                    } else {
                        valLossCursor = mx::reshape(config.ValidationData->GetCursor(),
                            {static_cast<int>(approxDim), static_cast<int>(valDocs)});
                    }
                    auto valLoss = target.ComputeLoss(
                        valLossCursor,
                        config.ValidationData->GetTargets(),
                        config.ValidationData->GetWeights()
                    );
                    float valLossVal = valLoss.item<float>();

                    CATBOOST_INFO_LOG << "CatBoost-MLX: iter=" << iter
                        << " train_loss=" << lossVal
                        << " val_loss=" << valLossVal
                        << " time=" << iterMs << "ms"
                        << " total=" << elapsed << "s" << Endl;

                    // Early stopping check
                    if (config.EarlyStoppingPatience > 0) {
                        if (valLossVal < bestValLoss - 1e-7f) {
                            bestValLoss = valLossVal;
                            bestIteration = iter;
                            noImprovementCount = 0;
                        } else {
                            noImprovementCount++;
                            if (noImprovementCount >= config.EarlyStoppingPatience) {
                                CATBOOST_INFO_LOG << "CatBoost-MLX: Early stopping at iteration "
                                    << iter << " (best val_loss=" << bestValLoss
                                    << " at iter=" << bestIteration << ")" << Endl;
                                break;
                            }
                        }
                    }
                } else {
                    CATBOOST_INFO_LOG << "CatBoost-MLX: iter=" << iter
                        << " loss=" << lossVal
                        << " time=" << iterMs << "ms"
                        << " total=" << elapsed << "s" << Endl;
                }
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
        result.BestIteration = bestIteration;

        auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        CATBOOST_INFO_LOG << "CatBoost-MLX: Training complete. "
            << result.NumIterations << " trees in " << totalTime << "s" << Endl;

        return result;
    }

}  // namespace NCatboostMlx
