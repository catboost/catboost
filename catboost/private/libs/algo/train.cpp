#include "train.h"

#include "approx_calcer.h"
#include "approx_calcer_helpers.h"
#include "approx_updater_helpers.h"
#include "build_subset_in_leaf.h"
#include "fold.h"
#include "greedy_tensor_search.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "monotonic_constraint_utils.h"
#include "online_ctr.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/helpers/query_info_helper.h>
#include <catboost/libs/logging/profile_info.h>
#include <catboost/private/libs/algo/approx_calcer/leafwise_approx_calcer.h>
#include <catboost/private/libs/algo_helpers/approx_calcer_helpers.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/distributed/master.h>
#include <catboost/private/libs/distributed/worker.h>


TErrorTracker BuildErrorTracker(
    EMetricBestValue bestValueType,
    double bestPossibleValue,
    bool hasTest,
    const TLearnContext& ctx
) {
    const auto& odOptions = ctx.Params.BoostingOptions->OverfittingDetector;
    return CreateErrorTracker(odOptions, bestPossibleValue, bestValueType, hasTest);
}

static void UpdateLearningFold(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& bestTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        data,
        error,
        *fold,
        bestTree,
        randomSeed,
        ctx,
        &approxDelta
    );

    if (error.GetIsExpApprox()) {
        UpdateBodyTailApprox</*StoreExpApprox*/true>(
            approxDelta,
            ctx->Params.BoostingOptions->LearningRate,
            ctx->LocalExecutor,
            fold
        );
    } else {
        UpdateBodyTailApprox</*StoreExpApprox*/false>(
            approxDelta,
            ctx->Params.BoostingOptions->LearningRate,
            ctx->LocalExecutor,
            fold
        );
    }
}

static void ScaleAllApproxes(
    const double approxMultiplier,
    const bool storeExpApprox,
    TLearnProgress* learnProgress,
    NPar::ILocalExecutor* localExecutor
) {
    TVector<TVector<TVector<double>>*> allApproxes;
    for (auto& fold : learnProgress->Folds) {
        for (auto &bodyTail : fold.BodyTailArr) {
            allApproxes.push_back(&bodyTail.Approx);
        }
    }
    allApproxes.push_back(&learnProgress->AveragingFold.BodyTailArr[0].Approx);
    const int learnApproxesCount = SafeIntegerCast<int>(allApproxes.size());
    allApproxes.push_back(&learnProgress->AvrgApprox);
    for (auto& testApprox : learnProgress->TestApprox) {
        allApproxes.push_back(&testApprox);
    }

    NPar::ParallelFor(
        *localExecutor,
        0,
        allApproxes.size(),
        [approxMultiplier, storeExpApprox, learnApproxesCount, localExecutor, &allApproxes](int index) {
            const bool isLearnApprox = (index < learnApproxesCount);
            if (isLearnApprox && storeExpApprox) {
                UpdateApprox(
                    [approxMultiplier](TConstArrayRef<double> /* delta */, TArrayRef<double> approx, size_t idx) {
                        approx[idx] = ApplyLearningRate<true>(approx[idx], approxMultiplier);
                    },
                    *allApproxes[index], // stub deltas
                    allApproxes[index],
                    localExecutor
                );
            } else {
                UpdateApprox(
                    [approxMultiplier](TConstArrayRef<double> /* delta */, TArrayRef<double> approx, size_t idx) {
                        approx[idx] = ApplyLearningRate<false>(approx[idx], approxMultiplier);
                    },
                    *allApproxes[index], // stub deltas
                    allApproxes[index],
                    localExecutor
                );
            }
        }
    );
}

static void CalcApproxesLeafwise(
    const NCB::TTrainingDataProviders& data,
    const IDerCalcer& error,
    const std::variant<TSplitTree, TNonSymmetricTreeStructure>& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues,
    TVector<TIndexType>* indices
) {
    *indices = BuildIndices(
        ctx->LearnProgress->AveragingFold,
        tree,
        data,
        EBuildIndicesDataParts::All,
        ctx->LocalExecutor
    );
    auto statistics = BuildSubset(
        *indices,
        GetLeafCount(tree),
        ctx
    );

    TVector<TDers> weightedDers;
    const int approxDimension = ctx->LearnProgress->AveragingFold.GetApproxDimension();
    if (approxDimension == 1) {
        const int scratchSize = APPROX_BLOCK_SIZE * CB_THREAD_LIMIT;
        weightedDers.yresize(scratchSize);
    }
    for (int leafIdx = 0; leafIdx < GetLeafCount(tree); ++leafIdx) {
        CalcLeafValues(
            error,
            &(statistics[leafIdx]),
            ctx,
            weightedDers
        );
    }
    AssignLeafValues(
        statistics,
        treeValues
    );

    // cycle for accordance with non leawfise approxes
    if (ctx->LearnProgress->AveragingFold.BodyTailArr[0].Approx.size() < 2
        && ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod != ELeavesEstimation::Exact
    ) {
        ctx->LearnProgress->Rand.Advance(ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations);
    }

}

void TrainOneIteration(const NCB::TTrainingDataProviders& data, TLearnContext* ctx) {
    const auto error = BuildError(ctx->Params, ctx->ObjectiveDescriptor);
    ctx->LearnProgress->HessianType = error->GetHessianType();
    TProfileInfo& profile = ctx->Profile;

    const size_t iterationIndex = ctx->LearnProgress->TreeStruct.size();
    const int foldCount = ctx->LearnProgress->Folds.ysize();
    const double modelLength
        = double(iterationIndex) * ctx->Params.BoostingOptions->LearningRate;

    CheckInterrupted(); // check after long-lasting operation

    const double modelShrinkRate = ctx->Params.BoostingOptions->ModelShrinkRate.Get();
    if (modelShrinkRate > 0) {
        if (iterationIndex > 0) {
            const double modelShrinkage =
                ctx->Params.BoostingOptions->ModelShrinkMode == EModelShrinkMode::Constant
                ? (1 - modelShrinkRate * ctx->Params.BoostingOptions->LearningRate)
                : (1 - modelShrinkRate / static_cast<double>(iterationIndex));
            ScaleAllApproxes(
                modelShrinkage,
                error->GetIsExpApprox(),
                ctx->LearnProgress.Get(),
                ctx->LocalExecutor
            );
            if (ctx->LearnProgress->StartingApprox.Defined()) {
                for (auto& approx : *ctx->LearnProgress->StartingApprox) {
                    approx = approx * modelShrinkage;
                }
            }
            ctx->LearnProgress->ModelShrinkHistory.push_back(modelShrinkage);
        } else {
            ctx->LearnProgress->ModelShrinkHistory.push_back(1.0);
        }
    }

    std::variant<TSplitTree, TNonSymmetricTreeStructure> bestTree;
    {
        TFold* takenFold = &ctx->LearnProgress->Folds[ctx->LearnProgress->Rand.GenRand() % foldCount];
        const TVector<ui64> randomSeeds = GenRandUI64Vector(
            takenFold->BodyTailArr.ysize(),
            ctx->LearnProgress->Rand.GenRand()
        );
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            ctx->LocalExecutor->ExecRangeWithThrow(
                [&](int bodyTailId) {
                    CalcWeightedDerivatives(
                        *error,
                        bodyTailId,
                        ctx->Params,
                        randomSeeds[bodyTailId],
                        takenFold,
                        ctx->LocalExecutor
                    );
                },
                0,
                takenFold->BodyTailArr.ysize(),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
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
            &bestTree
        );
    }
    CheckInterrupted(); // check after long-lasting operation
    {
        TVector<TFold*> trainFolds;
        for (int foldId = 0; foldId < foldCount; ++foldId) {
            trainFolds.push_back(&ctx->LearnProgress->Folds[foldId]);
        }

        TrimOnlineCTRcache(trainFolds);
        TrimOnlineCTRcache({ &ctx->LearnProgress->AveragingFold });
        {
            TVector<TFold*> allFolds = trainFolds;
            allFolds.push_back(&ctx->LearnProgress->AveragingFold);

            struct TLocalJobData {
                const NCB::TTrainingDataProviders* data;
                TProjection Projection;
                TFold* Fold;
                TOwnedOnlineCtr* Ctr;

            public:
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*data, *Fold, Projection, ctx, Ctr);
                }
            };

            TVector<TLocalJobData> parallelJobsData;
            THashSet<TProjection> seenProjections;
            for (const auto& ctr : GetUsedCtrs(bestTree)) {
                const auto& proj = ctr.Projection;
                if (seenProjections.contains(proj)) {
                    continue;
                }
                for (auto* foldPtr : allFolds) {
                    auto* ownedCtrs = foldPtr->GetOwnedCtrs(proj);
                    if (ownedCtrs && ownedCtrs->Data[proj].Feature.empty()) {
                        parallelJobsData.emplace_back(
                            TLocalJobData{ &data, proj, foldPtr, ownedCtrs}
                        );
                    }
                }
                seenProjections.insert(proj);
            }

            ctx->LocalExecutor->ExecRange(
                [&](int taskId){
                    parallelJobsData[taskId].DoTask(ctx);
                },
                0,
                SafeIntegerCast<int>(parallelJobsData.size()),
                NPar::TLocalExecutor::WAIT_COMPLETE
            );
        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        TVector<double> sumLeafWeights; // [leafId]

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            const TVector<ui64> randomSeeds = GenRandUI64Vector(foldCount, ctx->LearnProgress->Rand.GenRand());

            TVector<TIndexType> indices;

            const bool treeHasMonotonicConstraints = !ctx->Params.ObliviousTreeOptions->MonotoneConstraints.GetUnchecked().empty();
            if (
                ctx->Params.ObliviousTreeOptions->DevLeafwiseApproxes.Get() &&
                ctx->Params.BoostingOptions->BoostingType.Get() == EBoostingType::Plain
                && !treeHasMonotonicConstraints
                && error->GetErrorType() == EErrorType::PerObjectError
            ) {
                CalcApproxesLeafwise(
                    data,
                    *error,
                    bestTree,
                    ctx,
                    &treeValues,
                    &indices
                );
            } else {
                CalcLeafValues(
                    data,
                    *error,
                    bestTree,
                    ctx,
                    &treeValues,
                    &indices
                );
            }

            ctx->Profile.AddOperation("CalcApprox result leaves");
            CheckInterrupted(); // check after long-lasting operation

            TConstArrayRef<ui32> learnPermutationRef = ctx->LearnProgress->AveragingFold.GetLearnPermutationArray();

            const size_t leafCount = treeValues[0].size();
            sumLeafWeights = SumLeafWeights(
                leafCount,
                indices,
                learnPermutationRef,
                GetWeights(*data.Learn->TargetData),
                ctx->LocalExecutor
            );
            const auto lossFunction = ctx->Params.LossFunctionDescription->GetLossFunction();
            const bool usePairs = UsesPairsForCalculation(lossFunction);
            NormalizeLeafValues(
                usePairs,
                ctx->Params.BoostingOptions->LearningRate,
                sumLeafWeights,
                &treeValues
            );

            TVector<TVector<double>>* foldZeroApprox = nullptr;
            if (UseAveragingFoldAsFoldZero(*ctx)) {
                foldZeroApprox = &trainFolds[0]->BodyTailArr[0].Approx;
            }
            UpdateAvrgApprox(
                error->GetIsExpApprox(),
                data.Learn->GetObjectCount(),
                indices,
                treeValues,
                data.Test,
                ctx->LearnProgress.Get(),
                ctx->LocalExecutor,
                foldZeroApprox);
            ctx->LocalExecutor->ExecRangeWithThrow(
                [&](int foldId)
                {
                    UpdateLearningFold(
                        data,
                        *error,
                        bestTree,
                        randomSeeds[foldId],
                        trainFolds[foldId],
                        ctx);
                },
                /*firstId=*/static_cast<int>(foldZeroApprox != nullptr),
                foldCount,
                NPar::TLocalExecutor::WAIT_COMPLETE);
            profile.AddOperation("CalcApprox tree struct and update tree structure approx");
            CheckInterrupted(); // check after long-lasting operation
        } else {
            const bool isMultiTarget = dynamic_cast<const TMultiDerCalcer*>(error.Get()) != nullptr;

            if (isMultiTarget || (ctx->LearnProgress->ApproxDimension != 1)) {
                MapSetApproxesMulti(*error, bestTree, &treeValues, &sumLeafWeights, ctx);
            } else {
                MapSetApproxesSimple(*error, bestTree, &treeValues, &sumLeafWeights, ctx);
            }
        }

        ctx->LearnProgress->TreeStats.emplace_back();
        ctx->LearnProgress->TreeStats.back().LeafWeightsSum = std::move(sumLeafWeights);
        ctx->LearnProgress->LeafValues.push_back(std::move(treeValues));
        ctx->LearnProgress->TreeStruct.push_back(std::move(bestTree));

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}
