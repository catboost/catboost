#include "train.h"

#include "approx_calcer.h"
#include "error_functions.h"
#include "fold.h"
#include "greedy_tensor_search.h"
#include "online_ctr.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/data_new/data_provider.h>
#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/logging/profile_info.h>

TErrorTracker BuildErrorTracker(EMetricBestValue bestValueType, double bestPossibleValue, bool hasTest, const TLearnContext& ctx) {
    const auto& odOptions = ctx.Params.BoostingOptions->OverfittingDetector;
    return CreateErrorTracker(odOptions, bestPossibleValue, bestValueType, hasTest);
}

static void UpdateLearningFold(
    const NCB::TTrainingForCPUDataProviders& data,
    const IDerCalcer& error,
    const TSplitTree& bestSplitTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        data,
        error,
        *fold,
        bestSplitTree,
        randomSeed,
        ctx,
        &approxDelta
    );

    if (error.GetIsExpApprox()) {
        UpdateBodyTailApprox</*StoreExpApprox*/ true>(approxDelta, ctx->Params.BoostingOptions->LearningRate, ctx->LocalExecutor, fold);
    } else {
        UpdateBodyTailApprox</*StoreExpApprox*/ false>(approxDelta, ctx->Params.BoostingOptions->LearningRate, ctx->LocalExecutor, fold);
    }
}

void TrainOneIteration(const NCB::TTrainingForCPUDataProviders& data, TLearnContext* ctx) {
    const auto error = BuildError(ctx->Params, ctx->ObjectiveDescriptor);
    ctx->LearnProgress.HessianType = error->GetHessianType();
    CheckDerivativeOrderForTrain(
        error->GetMaxSupportedDerivativeOrder(),
        ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod
    );
    TProfileInfo& profile = ctx->Profile;

    const int foldCount = ctx->LearnProgress.Folds.ysize();
    const int currentIteration = ctx->LearnProgress.TreeStruct.ysize();
    const double modelLength = currentIteration * ctx->Params.BoostingOptions->LearningRate;

    CheckInterrupted(); // check after long-lasting operation

    TSplitTree bestSplitTree;
    {
        TFold* takenFold = &ctx->LearnProgress.Folds[ctx->Rand.GenRand() % foldCount];
        const TVector<ui64> randomSeeds = GenRandUI64Vector(takenFold->BodyTailArr.ysize(), ctx->Rand.GenRand());
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            ctx->LocalExecutor->ExecRange([&](int bodyTailId) {
                CalcWeightedDerivatives(*error, bodyTailId, ctx->Params, randomSeeds[bodyTailId], takenFold, ctx->LocalExecutor);
            }, 0, takenFold->BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            Y_ASSERT(takenFold->BodyTailArr.ysize() == 1);
            MapSetDerivatives(ctx);
        }
        profile.AddOperation("Calc derivatives");

        GreedyTensorSearch(
            data,
            modelLength,
            profile,
            takenFold,
            ctx,
            &bestSplitTree
        );
    }
    CheckInterrupted(); // check after long-lasting operation
    {
        TVector<TFold*> trainFolds;
        for (int foldId = 0; foldId < foldCount; ++foldId) {
            trainFolds.push_back(&ctx->LearnProgress.Folds[foldId]);
        }

        TrimOnlineCTRcache(trainFolds);
        TrimOnlineCTRcache({ &ctx->LearnProgress.AveragingFold });
        {
            TVector<TFold*> allFolds = trainFolds;
            allFolds.push_back(&ctx->LearnProgress.AveragingFold);

            struct TLocalJobData {
                const NCB::TTrainingForCPUDataProviders* data;
                TProjection Projection;
                TFold* Fold;
                TOnlineCTR* Ctr;
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*data, *Fold, Projection, ctx, Ctr);
                }
            };

            TVector<TLocalJobData> parallelJobsData;
            THashSet<TProjection> seenProjections;
            for (const auto& split : bestSplitTree.Splits) {
                if (split.Type != ESplitType::OnlineCtr) {
                    continue;
                }

                const auto& proj = split.Ctr.Projection;
                if (seenProjections.contains(proj)) {
                    continue;
                }
                for (auto* foldPtr : allFolds) {
                    if (!foldPtr->GetCtrs(proj).contains(proj) || foldPtr->GetCtr(proj).Feature.empty()) {
                        parallelJobsData.emplace_back(TLocalJobData{ &data, proj, foldPtr, &foldPtr->GetCtrRef(proj) });
                    }
                }
                seenProjections.insert(proj);
            }

            ctx->LocalExecutor->ExecRange([&](int taskId){
                parallelJobsData[taskId].DoTask(ctx);
            }, 0, parallelJobsData.size(), NPar::TLocalExecutor::WAIT_COMPLETE);

        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        TVector<double> sumLeafWeights; // [leafId]

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            const TVector<ui64> randomSeeds = GenRandUI64Vector(foldCount, ctx->Rand.GenRand());
            ctx->LocalExecutor->ExecRange([&](int foldId) {
                UpdateLearningFold(data, *error, bestSplitTree, randomSeeds[foldId], trainFolds[foldId], ctx);
            }, 0, foldCount, NPar::TLocalExecutor::WAIT_COMPLETE);

            profile.AddOperation("CalcApprox tree struct and update tree structure approx");
            CheckInterrupted(); // check after long-lasting operation

            TVector<TIndexType> indices;
            CalcLeafValues(
                data,
                *error,
                ctx->LearnProgress.AveragingFold,
                bestSplitTree,
                ctx,
                &treeValues,
                &indices
            );

            ctx->Profile.AddOperation("CalcApprox result leaves");
            CheckInterrupted(); // check after long-lasting operation

            TConstArrayRef<ui32> learnPermutationRef = ctx->LearnProgress.AveragingFold.GetLearnPermutationArray();

            const size_t leafCount = treeValues[0].size();
            sumLeafWeights = SumLeafWeights(leafCount, indices, learnPermutationRef, GetWeights(data.Learn->TargetData));
            NormalizeLeafValues(
                UsesPairsForCalculation(ctx->Params.LossFunctionDescription->GetLossFunction()),
                ctx->Params.BoostingOptions->LearningRate,
                sumLeafWeights,
                &treeValues
            );

            UpdateAvrgApprox(error->GetIsExpApprox(), data.Learn->GetObjectCount(), indices, treeValues, data.Test, &ctx->LearnProgress, ctx->LocalExecutor);
        } else {
            if (ctx->LearnProgress.ApproxDimension == 1) {
                MapSetApproxesSimple(*error, bestSplitTree, data.Test, &treeValues, &sumLeafWeights, ctx);
            } else {
                MapSetApproxesMulti(*error, bestSplitTree, data.Test, &treeValues, &sumLeafWeights, ctx);
            }
        }

        ctx->LearnProgress.TreeStats.emplace_back();
        ctx->LearnProgress.TreeStats.back().LeafWeightsSum = sumLeafWeights;
        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}
