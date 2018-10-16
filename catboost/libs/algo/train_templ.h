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

template <typename TError>
void UpdateLearningFold(
    const TDataset& learnData,
    const TDatasetPtrs& testDataPtrs,
    const TError& error,
    const TSplitTree& bestSplitTree,
    ui64 randomSeed,
    TFold* fold,
    TLearnContext* ctx
) {
    TVector<TVector<TVector<double>>> approxDelta;

    CalcApproxForLeafStruct(
        learnData,
        testDataPtrs,
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
    const TDatasetPtrs& testDataPtrs,
    const TError& error,
    const TSplitTree& bestSplitTree,
    TLearnContext* ctx,
    TVector<TVector<double>>* treeValues
) {
    TProfileInfo& profile = ctx->Profile;
    TVector<TIndexType> indices;

    CalcLeafValues(
        learnData,
        testDataPtrs,
        error,
        ctx->LearnProgress.AveragingFold,
        bestSplitTree,
        ctx,
        treeValues,
        &indices
    );

    const size_t leafCount = (*treeValues)[0].size();
    ctx->LearnProgress.TreeStats.emplace_back();
    ctx->LearnProgress.TreeStats.back().LeafWeightsSum = SumLeafWeights(leafCount, indices, ctx->LearnProgress.AveragingFold.LearnPermutation, learnData.Weights);

    if (IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction())) {
        const auto& leafWeightsSum = ctx->LearnProgress.TreeStats.back().LeafWeightsSum;
        double averageLeafValue = 0;
        for (size_t leafIdx : xrange(leafWeightsSum.size())) {
            averageLeafValue += (*treeValues)[0][leafIdx] * leafWeightsSum[leafIdx];
        }
        averageLeafValue /= Accumulate(leafWeightsSum, /*val*/0.0);
        for (auto& leafValue : (*treeValues)[0]) {
            leafValue -= averageLeafValue;
        }
    }

    const int approxDimension = ctx->LearnProgress.AvrgApprox.ysize();
    const double learningRate = ctx->Params.BoostingOptions->LearningRate;

    TVector<TVector<double>> expTreeValues;
    expTreeValues.yresize(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (auto& leafVal : (*treeValues)[dim]) {
            leafVal *= learningRate;
        }
        expTreeValues[dim] = (*treeValues)[dim];
        ExpApproxIf(TError::StoreExpApprox, &expTreeValues[dim]);
    }

    profile.AddOperation("CalcApprox result leaves");
    CheckInterrupted(); // check after long-lasting operation

    Y_ASSERT(ctx->LearnProgress.AveragingFold.BodyTailArr.ysize() == 1);
    const size_t learnSampleCount = learnData.GetSampleCount();
    const TVector<size_t>& testOffsets = CalcTestOffsets(learnSampleCount, testDataPtrs);

    ctx->LocalExecutor.ExecRange([&](int setIdx){
        if (setIdx == 0) { // learn data set
            TConstArrayRef<TIndexType> indicesRef(indices);
            const auto updateApprox = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
                approx[idx] = UpdateApprox<TError::StoreExpApprox>(approx[idx], delta[indicesRef[idx]]);
            };
            TFold::TBodyTail& bt = ctx->LearnProgress.AveragingFold.BodyTailArr[0];
            Y_ASSERT(bt.Approx[0].ysize() == bt.TailFinish);
            UpdateApprox(updateApprox, expTreeValues, &bt.Approx, &ctx->LocalExecutor);

            TConstArrayRef<ui32> learnPermutationRef(ctx->LearnProgress.AveragingFold.LearnPermutation);
            const auto updateAvrgApprox = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
                approx[learnPermutationRef[idx]] += delta[indicesRef[idx]];
            };
            Y_ASSERT(ctx->LearnProgress.AvrgApprox[0].size() == learnSampleCount);
            UpdateApprox(updateAvrgApprox, *treeValues, &ctx->LearnProgress.AvrgApprox, &ctx->LocalExecutor);
        } else { // test data set
            const int testIdx = setIdx - 1;
            const size_t testSampleCount = testDataPtrs[testIdx]->GetSampleCount();
            TConstArrayRef<TIndexType> indicesRef(indices.data() + testOffsets[testIdx], testSampleCount);
            const auto updateTestApprox = [=](TConstArrayRef<double> delta, TArrayRef<double> approx, size_t idx) {
                approx[idx] += delta[indicesRef[idx]];
            };
            Y_ASSERT(ctx->LearnProgress.TestApprox[testIdx][0].size() == testSampleCount);
            UpdateApprox(updateTestApprox, *treeValues, &ctx->LearnProgress.TestApprox[testIdx], &ctx->LocalExecutor);
        }
    }, 0, 1 + testDataPtrs.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <typename TError>
void TrainOneIter(const TDataset& learnData, const TDatasetPtrs& testDataPtrs, TLearnContext* ctx) {
    ctx->LearnProgress.HessianType = TError::GetHessianType();
    TError error = BuildError<TError>(ctx->Params, ctx->ObjectiveDescriptor);
    CheckDerivativeOrderForTrain(
        error.GetMaxSupportedDerivativeOrder(),
        ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod
    );
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
            testDataPtrs,
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
                const TDatasetPtrs& TestDatas;
                TProjection Projection;
                TFold* Fold;
                TOnlineCTR* Ctr;
                void DoTask(TLearnContext* ctx) {
                    ComputeOnlineCTRs(*LearnData, TestDatas, *Fold, Projection, ctx, Ctr);
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
                        parallelJobsData.emplace_back(TLocalJobData{ &learnData, testDataPtrs, proj, foldPtr, &foldPtr->GetCtrRef(proj) });
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
                UpdateLearningFold(learnData, testDataPtrs, error, bestSplitTree, randomSeeds[foldId], trainFolds[foldId], ctx);
            }, 0, foldCount, NPar::TLocalExecutor::WAIT_COMPLETE);
        } else {
            if (ctx->LearnProgress.AveragingFold.GetApproxDimension() == 1) {
                MapSetApproxesSimple<TError>(bestSplitTree, ctx);
            } else {
                MapSetApproxesMulti<TError>(bestSplitTree, ctx);
            }
        }

        profile.AddOperation("CalcApprox tree struct and update tree structure approx");
        CheckInterrupted(); // check after long-lasting operation

        TVector<TVector<double>> treeValues; // [dim][leafId]
        UpdateAveragingFold(learnData, testDataPtrs, error, bestSplitTree, ctx, &treeValues);

        ctx->LearnProgress.LeafValues.push_back(treeValues);
        ctx->LearnProgress.TreeStruct.push_back(bestSplitTree);

        profile.AddOperation("Update final approxes");
        CheckInterrupted(); // check after long-lasting operation
    }
}
