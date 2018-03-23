#pragma once

#include "approx_calcer.h"
#include "fold.h"
#include "greedy_tensor_search.h"
#include "online_ctr.h"
#include "tensor_search_helpers.h"

#include <catboost/libs/distributed/worker.h>
#include <catboost/libs/distributed/master.h>
#include <catboost/libs/helpers/interrupt.h>
#include <catboost/libs/logging/profile_info.h>

struct TCompetitor;

namespace {
void NormalizeLeafValues(const TVector<TIndexType>& indices, int learnSampleCount, TVector<TVector<double>>* treeValues) {
    TVector<int> weights((*treeValues)[0].ysize());
    for (int docIdx = 0; docIdx < learnSampleCount; ++docIdx) {
        ++weights[indices[docIdx]];
    }

    double avrg = 0;
    for (int i = 0; i < weights.ysize(); ++i) {
        avrg += weights[i] * (*treeValues)[0][i];
    }

    int sumWeight = 0;
    for (int w : weights) {
        sumWeight += w;
    }
    avrg /= sumWeight;

    for (auto& value : (*treeValues)[0]) {
        value -= avrg;
    }
}
}

template <typename TError>
void UpdateLearningFold(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TSplitTree& bestSplitTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        learnData,
        testData,
        error,
        *fold,
        bestSplitTree,
        randomSeed,
        ctx,
        &approxDelta
    );

    UpdateBodyTailApprox<TError::StoreExpApprox>(approxDelta, ctx->Params.BoostingOptions->LearningRate, &ctx->LocalExecutor, fold);
}

template <typename TError>
void UpdateAveragingFold(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TSplitTree& bestSplitTree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues
) {
    TProfileInfo& profile = ctx->Profile;
    TVector<TIndexType> indices;

    CalcLeafValues(
        learnData,
        testData,
        error,
        ctx->LearnProgress.AveragingFold,
        bestSplitTree,
        ctx,
        treeValues,
        &indices
    );
    auto& currentTreeStats = ctx->LearnProgress.TreeStats.emplace_back();
    currentTreeStats.LeafWeightsSum.resize((*treeValues)[0].size());
    for (auto docId = 0; docId < learnData.GetSampleCount(); ++docId) {
        currentTreeStats.LeafWeightsSum[indices[ctx->LearnProgress.AveragingFold.LearnPermutation[docId]]] += learnData.Weights[docId];
    }
    // TODO(nikitxskv): if this will be a bottleneck, we can use precalculated counts.
    if (IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        NormalizeLeafValues(indices, learnData.GetSampleCount(), treeValues);
    }

    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
    const double learningRate = ctx->Params.BoostingOptions->LearningRate;
    const auto sampleCount = learnData.GetSampleCount() + (testData ? testData->GetSampleCount() : 0);

    TVector<TVector<double>> expTreeValues;
    expTreeValues.yresize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (auto& leafVal : (*treeValues)[dim]) {
            leafVal *= learningRate;
        }
        expTreeValues[dim] = (*treeValues)[dim];
        ExpApproxIf(TError::StoreExpApprox, &expTreeValues[dim]);
    }

    profile.AddOperation("CalcApprox result leafs");
    CheckInterrupted(); // check after long-lasting operation

    Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
    TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];

    const int tailFinish = bt.TailFinish;
    const int learnSampleCount = learnData.GetSampleCount();
    const size_t* learnPermutationData = ctx->LearnProgress.AveragingFold.LearnPermutation.data();
    const TIndexType* indicesData = indices.data();
    for (int dim = 0; dim < approxDimension; ++dim) {
        const double* expTreeValuesData = expTreeValues[dim].data();
        const double* treeValuesData = (*treeValues)[dim].data();
        double* approxData = bt.Approx[dim].data();
        double* avrgApproxData = ctx->LearnProgress.AvrgApprox[dim].data();
        double* testApproxData = ctx->LearnProgress.TestApprox[dim].data();
        ctx->LocalExecutor.ExecRange(
            [=](int docIdx) {
                const int permutedDocIdx = docIdx < learnSampleCount ? learnPermutationData[docIdx] : docIdx;
                if (docIdx < tailFinish) {
                    Y_VERIFY(docIdx < learnSampleCount);
                    approxData[docIdx] = UpdateApprox<TError::StoreExpApprox>(approxData[docIdx], expTreeValuesData[indicesData[docIdx]]);
                }
                if (docIdx < learnSampleCount) {
                    avrgApproxData[permutedDocIdx] += treeValuesData[indicesData[docIdx]];
                } else {
                    testApproxData[docIdx - learnSampleCount] += treeValuesData[indicesData[docIdx]];
                }
            },
            NPar::TLocalExecutor::TExecRangeParams(0, sampleCount).SetBlockSize(1000),
            NPar::TLocalExecutor::WAIT_COMPLETE
        );
    }
}

template <typename TError>
void TrainOneIter(const TDataset& learnData, const TDataset* testData, TLearnContext* ctx) {
    TError error = BuildError<TError>(ctx->Params, ctx->ObjectiveDescriptor);
    TProfileInfo& profile = ctx->Profile;

    const TVector<int> splitCounts = CountSplits(ctx->LearnProgress.FloatFeatures);

    const int foldCount = ctx->LearnProgress.Folds.ysize();
    const int currentIteration = ctx->LearnProgress.TreeStruct.ysize();
    const double modelLength = currentIteration * ctx->Params.BoostingOptions->LearningRate;

    CheckInterrupted(); // check after long-lasting operation

    TSplitTree bestSplitTree;
    {
        TFold* takenFold = &ctx->LearnProgress.Folds[ctx->Rand.GenRand() % foldCount];
        const TVector<ui64> randomSeeds = GenRandUI64Vector(takenFold->BodyTailArr.ysize(), ctx->Rand.GenRand());
        if (ctx->Params.SystemOptions->IsSingleHost()) {
            ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
                CalcWeightedDerivatives(error, bodyTailId, ctx->Params, randomSeeds[bodyTailId], takenFold, &ctx->LocalExecutor);
            }, 0, takenFold->BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            Y_ASSERT(takenFold->BodyTailArr.ysize() == 1);
            MapSetDerivatives<TError>(ctx);
        }
        profile.AddOperation("Calc derivatives");

        GreedyTensorSearch(
            learnData,
            testData,
            splitCounts,
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
                const TDataset* LearnData;
                const TDataset* TestData;
                TProjection Projection;
                TFold* Fold;
                TOnlineCTR* Ctr;
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*LearnData, TestData, *Fold, Projection, ctx, Ctr);
                }
            };

            TVector<TLocalJobData> parallelJobsData;
            THashSet<TProjection> seenProjections;
            for (const auto& split : bestSplitTree.Splits) {
                if (split.Type != ESplitType::OnlineCtr) {
                    continue;
                }

                const auto& proj = split.Ctr.Projection;
                if (seenProjections.has(proj)) {
                    continue;
                }
                for (auto* foldPtr : allFolds) {
                    if (!foldPtr->GetCtrs(proj).has(proj) || foldPtr->GetCtr(proj).Feature.empty()) {
                        parallelJobsData.emplace_back(TLocalJobData{ &learnData, testData, proj, foldPtr, &foldPtr->GetCtrRef(proj) });
                    }
                }
                seenProjections.insert(proj);
            }

            ctx->LocalExecutor.ExecRange([&](int taskId){
                parallelJobsData[taskId].DoTask(ctx);
            }, 0, parallelJobsData.size(), NPar::TLocalExecutor::WAIT_COMPLETE);

        }
        profile.AddOperation("ComputeOnlineCTRs for tree struct (train folds and test fold)");
        CheckInterrupted(); // check after long-lasting operation

        if (ctx->Params.SystemOptions->IsSingleHost()) {
            const TVector<ui64> randomSeeds = GenRandUI64Vector(foldCount, ctx->Rand.GenRand());
            ctx->LocalExecutor.ExecRange([&](int foldId) {
                UpdateLearningFold(learnData, testData, error, bestSplitTree, randomSeeds[foldId], trainFolds[foldId], ctx);
            }, 0, foldCount, NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            MapSetApproxes<TError>(bestSplitTree, ctx);
        }

        profile.AddOperation("CalcApprox tree struct and update tree structure approx");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        UpdateAveragingFold(learnData, testData, error, bestSplitTree, ctx, &treeValues);

        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}
