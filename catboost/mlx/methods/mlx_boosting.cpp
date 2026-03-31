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
        // Cursor/gradients/hessians have shape [approxDim, numDocs].
        // For approxDim=1, we reshape to [numDocs] for the target functions.
        // TODO(Phase 7): Support multi-dimensional approx (e.g. multiclass)
        CB_ENSURE(trainData.GetCursor().shape(0) == 1 || trainData.GetCursor().ndim() == 1,
            "CatBoost-MLX: Only single-dimensional approx is currently supported (approxDim=1)");

        for (ui32 iter = 0; iter < config.NumIterations; ++iter) {
            auto iterStart = std::chrono::steady_clock::now();

            // ----- Step 1: Compute gradients and hessians -----
            // Flatten from [1, numDocs] to [numDocs] for target functions
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
            const ui32 numLeaves = 1u << treeStructure.Splits.size();

            // Compute gradient and hessian sums per leaf from histograms
            // For RMSE: gradSum per leaf comes from the histogram
            // HessSum per leaf = count of docs in leaf (since RMSE hessian = 1)
            // TODO: Extract actual sums from histogram result
            // For now, use uniform estimates (will be corrected in integration)
            auto gradSums = mx::zeros({static_cast<int>(numLeaves)}, mx::float32);
            auto hessSums = mx::ones({static_cast<int>(numLeaves)}, mx::float32);
            hessSums = mx::multiply(hessSums, mx::array(static_cast<float>(numDocs) / numLeaves));

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
