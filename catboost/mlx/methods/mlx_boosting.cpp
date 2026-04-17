#include "mlx_boosting.h"
#include <catboost/mlx/methods/leaves/leaf_estimator.h>
#include <catboost/mlx/methods/stage_profiler.h>
#include <catboost/libs/logging/logging.h>

#include <chrono>
#include <random>
#include <numeric>
#include <algorithm>
#include <string>

#ifdef CATBOOST_MLX_STAGE_PROFILE
#include <ctime>
#include <filesystem>
#include <sstream>
#endif

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

        const bool isDepthwise   = (config.GrowPolicy == EGrowPolicy::Depthwise);
        const bool isLossguide   = (config.GrowPolicy == EGrowPolicy::Lossguide);
        result.GrowPolicy = config.GrowPolicy;

        const char* policyName = isDepthwise ? "Depthwise"
                               : isLossguide ? "Lossguide"
                               : "SymmetricTree";

#ifdef CATBOOST_MLX_STAGE_PROFILE
        // Instantiate profiler for this run. Passed by pointer to structure searcher
        // functions so they can record depth-level stage timings.
        TStageProfiler stageProfiler(static_cast<int>(config.NumIterations));
        TStageProfiler* profiler = &stageProfiler;
#else
        TStageProfiler* profiler = nullptr;
#endif

        CATBOOST_INFO_LOG << "CatBoost-MLX: Starting boosting with "
            << config.NumIterations << " iterations, lr=" << config.LearningRate
            << ", depth=" << config.MaxDepth << ", l2=" << config.L2RegLambda
            << ", grow_policy=" << policyName
            << (isLossguide ? (", max_leaves=" + std::to_string(config.MaxLeaves)) : "")
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
            TMLXDevice::EvalAtBoundary(valCursor);
            CATBOOST_INFO_LOG << "CatBoost-MLX: Validation set: " << valDocs << " docs" << Endl;
        }

        // Random number generator for subsampling
        std::mt19937 rng(config.RandomSeed);

        for (ui32 iter = 0; iter < config.NumIterations; ++iter) {
            auto iterStart = std::chrono::steady_clock::now();

#ifdef CATBOOST_MLX_STAGE_PROFILE
            profiler->BeginIter(static_cast<int>(iter));
#endif

            // Single sync per iteration: materialise the cursor updated by the
            // previous iteration's Apply* call.  This bounds the lazy MLX graph
            // depth to O(1 iteration), preventing unbounded memory accumulation
            // across iterations and capping Metal command-buffer compilation time.
            // All downstream ops in this iteration (ComputeDerivatives, histograms,
            // leaf estimation, Apply*) run without additional syncs until next iter.
            TMLXDevice::EvalAtBoundary({trainData.GetCursor()});

            // ----- Step 1: Compute gradients and hessians -----
            mx::array cursor;
            if (approxDim == 1) {
                cursor = mx::reshape(trainData.GetCursor(), {static_cast<int>(numDocs)});
            } else {
                cursor = mx::reshape(trainData.GetCursor(),
                    {static_cast<int>(approxDim), static_cast<int>(numDocs)});
            }
            {
                STAGE_TIMER(profiler, EStageId::Derivatives,
                    {trainData.GetGradients(), trainData.GetHessians()});
                target.ComputeDerivatives(
                    cursor,
                    trainData.GetTargets(),
                    trainData.GetWeights(),
                    trainData.GetGradients(),
                    trainData.GetHessians()
                );
            }

            // ----- Step 2: Search for best tree structure -----
            {
                STAGE_TIMER(profiler, EStageId::InitPartitions, {});
                trainData.InitPartitions(numDocs);
            }

            // Depthwise path: grow per-leaf splits (level-wise, per-leaf best split).
            // Oblivious (SymmetricTree) path: one split shared across all leaves per level.
            // Lossguide path: best-first leaf-wise growth with max_leaves budget.
            TDepthwiseTreeStructure  depthwiseStructure;
            TObliviousTreeStructure  obliviousStructure;
            TLossguideTreeStructure  lossguideStructure;
            ui32 actualDepth  = 0;
            ui32 actualLeaves = 0;

            if (isLossguide) {
                lossguideStructure = SearchLossguideTreeStructure(
                    trainData,
                    config.MaxLeaves,
                    config.L2RegLambda,
                    approxDim,
                    /*maxDepth=*/config.MaxDepth,
                    profiler
                );
                actualLeaves = lossguideStructure.NumLeaves;
                if (actualLeaves <= 1u) {
                    // No splits were made (no valid split for the root).
                    CATBOOST_WARNING_LOG << "CatBoost-MLX: No valid lossguide splits at iteration "
                        << iter << ", stopping early" << Endl;
                    break;
                }
            } else if (isDepthwise) {
                depthwiseStructure = SearchDepthwiseTreeStructure(
                    trainData,
                    config.MaxDepth,
                    config.L2RegLambda,
                    approxDim,
                    profiler
                );
                actualDepth = depthwiseStructure.Depth;
                if (actualDepth == 0) {
                    CATBOOST_WARNING_LOG << "CatBoost-MLX: No valid depthwise splits at iteration "
                        << iter << ", stopping early" << Endl;
                    break;
                }
                actualLeaves = 1u << actualDepth;
            } else {
                obliviousStructure = SearchTreeStructure(
                    trainData,
                    config.MaxDepth,
                    config.L2RegLambda,
                    approxDim,
                    profiler
                );
                actualDepth = static_cast<ui32>(obliviousStructure.Splits.size());
                if (actualDepth == 0) {
                    CATBOOST_WARNING_LOG << "CatBoost-MLX: No valid splits at iteration " << iter
                        << ", stopping early" << Endl;
                    break;
                }
                actualLeaves = 1u << actualDepth;
            }

            // ----- Step 3: Estimate leaf values (GPU-accelerated) -----
            const ui32 numLeaves = actualLeaves;

            // For lossguide: use LeafDocIds (dense leaf assignment) as the partition array.
            // For depthwise/oblivious: use the bit-encoded GetPartitions() array.
            const mx::array& partitionArray = isLossguide
                ? lossguideStructure.LeafDocIds
                : trainData.GetPartitions();

            // Stage 6: Leaf sums.
            mx::array gradSumsGPU, hessSumsGPU;
            {
                STAGE_TIMER(profiler, EStageId::LeafSums,
                    {trainData.GetGradients(), trainData.GetHessians()});
                ComputeLeafSumsGPU(
                    trainData.GetGradients(),
                    trainData.GetHessians(),
                    partitionArray,
                    numDocs, numLeaves, approxDim,
                    gradSumsGPU, hessSumsGPU
                );
            }

            // Stage 7: Leaf values (Newton step).
            mx::array leafValues;
            {
                STAGE_TIMER(profiler, EStageId::LeafValues, {gradSumsGPU, hessSumsGPU});
                if (approxDim == 1) {
                    // Scalar path: gradSumsGPU / hessSumsGPU are [numLeaves].
                    // Returns a lazy array; materialised by Apply* or the result store below.
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
                    // Apply* functions expect:
                    //   leafValues[leafIdx * approxDim + k]
                    auto flatLeafValues = ComputeLeafValues(gradSumsGPU, hessSumsGPU,
                        config.L2RegLambda, config.LearningRate);
                    // flatLeafValues shape: [approxDim * numLeaves]
                    auto dimByLeaf = mx::reshape(flatLeafValues,
                        {static_cast<int>(approxDim), static_cast<int>(numLeaves)});
                    // dimByLeaf shape: [approxDim, numLeaves]
                    leafValues = mx::transpose(dimByLeaf);
                    // leafValues shape: [numLeaves, approxDim]
                }
            }

            // ----- Step 4: Apply tree to predictions -----
            // Stage 8: Tree apply.
            {
                STAGE_TIMER(profiler, EStageId::TreeApply,
                    {trainData.GetCursor()});
                if (isLossguide) {
                    ApplyLossguideTree(trainData,
                        lossguideStructure.NodeSplitMap, lossguideStructure.LeafBfsIds,
                        lossguideStructure.LeafDocIds,
                        leafValues, numLeaves, approxDim);
                    if (hasValidation) {
                        // Recompute leaf assignments for validation data via BFS traversal.
                        auto valLeafIds = ComputeLeafIndicesLossguide(
                            config.ValidationData->GetCompressedIndex().GetCompressedData(),
                            lossguideStructure.NodeSplitMap, lossguideStructure.LeafBfsIds,
                            config.ValidationData->GetNumDocs(),
                            numLeaves
                        );
                        ApplyLossguideTree(*config.ValidationData,
                            lossguideStructure.NodeSplitMap, lossguideStructure.LeafBfsIds,
                            valLeafIds,
                            leafValues, numLeaves, approxDim);
                    }
                } else if (isDepthwise) {
                    ApplyDepthwiseTree(trainData, depthwiseStructure.NodeSplits,
                        leafValues, actualDepth, approxDim);
                    if (hasValidation) {
                        ApplyDepthwiseTree(*config.ValidationData, depthwiseStructure.NodeSplits,
                            leafValues, actualDepth, approxDim);
                    }
                } else {
                    ApplyObliviousTree(trainData, obliviousStructure.Splits, leafValues, approxDim);
                    if (hasValidation) {
                        ApplyObliviousTree(*config.ValidationData, obliviousStructure.Splits,
                            leafValues, approxDim);
                    }
                }
            }

            // ----- Step 5: Report metrics -----
            if (isLossguide) {
                result.LossguideTreeStructures.push_back(std::move(lossguideStructure));
            } else if (isDepthwise) {
                result.DepthwiseTreeStructures.push_back(std::move(depthwiseStructure));
            } else {
                result.TreeStructures.push_back(std::move(obliviousStructure));
            }
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

                // Stage 9: Loss eval — ComputeLoss + item<float>() readback.
                float lossVal = 0.0f;
                {
                    STAGE_TIMER(profiler, EStageId::LossEval, {});
                    auto loss = target.ComputeLoss(
                        lossCursor,
                        trainData.GetTargets(),
                        trainData.GetWeights()
                    );
                    lossVal = loss.item<float>();
                }

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
                    float valLossVal = 0.0f;
                    {
                        STAGE_TIMER(profiler, EStageId::LossEval, {});
                        auto valLoss = target.ComputeLoss(
                            valLossCursor,
                            config.ValidationData->GetTargets(),
                            config.ValidationData->GetWeights()
                        );
                        valLossVal = valLoss.item<float>();
                    }

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

#ifdef CATBOOST_MLX_STAGE_PROFILE
            profiler->EndIter();
#endif

            // Check for early stopping via callbacks
            if (callbacks && metricsHistory) {
                if (!callbacks->IsContinueTraining(*metricsHistory)) {
                    CATBOOST_INFO_LOG << "CatBoost-MLX: Early stopping at iteration " << iter << Endl;
                    break;
                }
            }
        }

        result.NumIterations = isLossguide
            ? static_cast<ui32>(result.LossguideTreeStructures.size())
            : isDepthwise
                ? static_cast<ui32>(result.DepthwiseTreeStructures.size())
                : static_cast<ui32>(result.TreeStructures.size());
        result.BestIteration = bestIteration;

        auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        CATBOOST_INFO_LOG << "CatBoost-MLX: Training complete. "
            << result.NumIterations << " trees in " << totalTime << "s" << Endl;

#ifdef CATBOOST_MLX_STAGE_PROFILE
        // Dump stage profile JSON if a path was configured.
        // Output path: CATBOOST_MLX_PROFILE_PATH env var, or
        //              .cache/profiling/sprint16/stage_times_<unix_timestamp>.json by default.
        {
            // Build output path.
            const char* envPath = std::getenv("CATBOOST_MLX_PROFILE_PATH");
            std::string outPath;
            if (envPath && *envPath) {
                outPath = envPath;
            } else {
                // Default: .cache/profiling/sprint16/stage_times_<timestamp>.json
                // The directory must exist; we create it if possible.
                const char* baseDir = ".cache/profiling/sprint16";
                try {
                    std::filesystem::create_directories(baseDir);
                } catch (...) { /* ignore */ }

                auto ts = static_cast<long long>(
                    std::chrono::duration_cast<std::chrono::seconds>(
                        std::chrono::steady_clock::now().time_since_epoch()
                    ).count()
                );
                outPath = std::string(baseDir) + "/stage_times_" +
                          std::to_string(ts) + ".json";
            }

            // Build meta JSON object.
            std::ostringstream metaOs;
            metaOs << "{"
                   << "\"build\":\"CATBOOST_MLX_STAGE_PROFILE=ON\","
                   << "\"num_iterations\":" << result.NumIterations << ","
                   << "\"grow_policy\":\"" << policyName << "\","
                   << "\"max_depth\":" << config.MaxDepth << ","
                   << "\"approx_dim\":" << approxDim << ","
                   << "\"num_docs\":" << numDocs
                   << "}";

            if (stageProfiler.WriteJson(outPath, metaOs.str())) {
                CATBOOST_INFO_LOG << "CatBoost-MLX: Stage profile written to " << outPath << Endl;
            }
        }
#endif

        return result;
    }

}  // namespace NCatboostMlx
