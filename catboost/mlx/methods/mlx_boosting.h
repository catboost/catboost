#pragma once

// CatBoost-MLX boosting loop.
// Orchestrates the full gradient boosting iteration cycle on Apple Silicon GPU.
//
// Reference: catboost/cuda/methods/doc_parallel_boosting.h (Fit() method)

#include <catboost/mlx/gpu_data/mlx_data_set.h>
#include <catboost/mlx/targets/target_func.h>
#include <catboost/mlx/methods/structure_searcher.h>
#include <catboost/mlx/methods/tree_applier.h>

#include <catboost/libs/loggers/catboost_logger_helpers.h>
#include <catboost/libs/train_lib/train_model.h>

namespace NCatboostMlx {

    // Result of the boosting process.
    struct TBoostingResult {
        TVector<TObliviousTreeStructure> TreeStructures;
        TVector<mx::array> TreeLeafValues;   // [numTrees] each [2^depth]
        ui32 NumIterations;
    };

    // Configuration for the boosting loop.
    struct TBoostingConfig {
        ui32 NumIterations = 1000;
        float LearningRate = 0.03f;
        ui32 MaxDepth = 6;
        float L2RegLambda = 3.0f;
        bool UseWeights = false;
    };

    // Run the gradient boosting loop.
    //
    // Each iteration:
    //   1. Compute gradients/hessians from current predictions (target function)
    //   2. Search for best tree structure (histogram + scoring)
    //   3. Estimate leaf values (Newton step)
    //   4. Apply tree to predictions
    //   5. Report metrics
    TBoostingResult RunBoosting(
        TMLXDataSet& trainData,
        const IMLXTargetFunc& target,
        const TBoostingConfig& config,
        ITrainingCallbacks* callbacks = nullptr,
        TMetricsAndTimeLeftHistory* metricsHistory = nullptr
    );

}  // namespace NCatboostMlx
