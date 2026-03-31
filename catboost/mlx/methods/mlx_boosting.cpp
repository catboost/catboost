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

        const ui32 numDocs = trainData.GetNumDocs();

        CATBOOST_INFO_LOG << "CatBoost-MLX: Starting boosting with "
            << config.NumIterations << " iterations, lr=" << config.LearningRate
            << ", depth=" << config.MaxDepth << ", l2=" << config.L2RegLambda
            << ", docs=" << numDocs << Endl;

        auto startTime = std::chrono::steady_clock::now();

        // Currently only single-dimensional approx is supported (e.g. RMSE).
        CB_ENSURE(trainData.GetCursor().shape(0) == 1 || trainData.GetCursor().ndim() == 1,
            "CatBoost-MLX: Only single-dimensional approx is currently supported (approxDim=1)");

        for (ui32 iter = 0; iter < config.NumIterations; ++iter) {
            auto iterStart = std::chrono::steady_clock::now();

            // ----- Step 1: Compute gradients and hessians -----
            auto cursor = mx::reshape(trainData.GetCursor(), {static_cast<int>(numDocs)});
            target.ComputeDerivatives(
                cursor,
                trainData.GetTargets(),
                trainData.GetWeights(),
                trainData.GetGradients(),
                trainData.GetHessians()
            );

            // ----- Step 2: Search for best tree structure -----
            // Reset partitions to single leaf for new tree
            trainData.InitPartitions(numDocs);

            auto treeStructure = SearchTreeStructure(
                trainData,
                config.MaxDepth,
                config.L2RegLambda
            );

            if (treeStructure.Splits.empty()) {
                CATBOOST_WARNING_LOG << "CatBoost-MLX: No valid splits at iteration " << iter
                    << ", stopping early" << Endl;
                break;
            }

            // ----- Step 3: Estimate leaf values -----
            // After SearchTreeStructure, partitions hold the final leaf assignments.
            const ui32 numLeaves = 1u << treeStructure.Splits.size();

            // Compute per-leaf gradient and hessian sums from partition assignments
            TMLXDevice::EvalNow({trainData.GetGradients(), trainData.GetHessians(), trainData.GetPartitions()});

            auto flatGrads = mx::reshape(trainData.GetGradients(), {static_cast<int>(numDocs)});
            auto flatHess = mx::reshape(trainData.GetHessians(), {static_cast<int>(numDocs)});
            TMLXDevice::EvalNow({flatGrads, flatHess});

            const float* gradsPtr = flatGrads.data<float>();
            const float* hessPtr = flatHess.data<float>();
            const uint32_t* partsPtr = trainData.GetPartitions().data<uint32_t>();

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

            auto leafValues = ComputeLeafValues(
                gradSums,
                hessSums,
                config.L2RegLambda,
                config.LearningRate
            );

            // ----- Step 4: Apply tree to predictions -----
            ApplyObliviousTree(trainData, treeStructure.Splits, leafValues);

            // ----- Step 5: Report metrics -----
            result.TreeStructures.push_back(std::move(treeStructure));
            result.TreeLeafValues.push_back(leafValues);

            auto iterEnd = std::chrono::steady_clock::now();
            auto iterMs = std::chrono::duration_cast<std::chrono::milliseconds>(iterEnd - iterStart).count();

            if (iter % 100 == 0 || iter == config.NumIterations - 1) {
                auto loss = target.ComputeLoss(
                    mx::reshape(trainData.GetCursor(), {static_cast<int>(numDocs)}),
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
