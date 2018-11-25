#pragma once

#include "mappers.h"

#include <catboost/libs/algo/learn_context.h>
#include <catboost/libs/algo/split.h>
#include <catboost/libs/algo/tensor_search_helpers.h>
#include <catboost/libs/data/dataset.h>

void InitializeMaster(TLearnContext* ctx);
void FinalizeMaster(TLearnContext* ctx);
void MapBuildPlainFold(const TDataset& trainData, TLearnContext* ctx);
void MapRestoreApproxFromTreeStruct(TLearnContext* ctx);
void MapTensorSearchStart(TLearnContext* ctx);
void MapBootstrap(TLearnContext* ctx);
void MapCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx);
void MapRemoteCalcScore(double scoreStDev, int depth, TCandidateList* candidateList, TLearnContext* ctx);
void MapPairwiseCalcScore(double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx);
void MapRemotePairwiseCalcScore(double scoreStDev, TCandidateList* candidateList, TLearnContext* ctx);
void MapSetIndices(const TCandidateInfo& bestSplitCandidate, TLearnContext* ctx);
int MapGetRedundantSplitIdx(TLearnContext* ctx);
void MapCalcErrors(TLearnContext* ctx);

template <typename TError>
void MapSetDerivatives(TLearnContext* ctx);

template <typename TMapper>
TVector<typename TMapper::TOutput> ApplyMapper(int workerCount, TObj<NPar::IEnvironment> environment, const typename TMapper::TInput& value = typename TMapper::TInput()) {
    NPar::TJobDescription job;
    TVector<typename TMapper::TInput> mapperInput(1);
    mapperInput[0] = value;
    NPar::Map(&job, new TMapper(), &mapperInput);
    job.SeparateResults(workerCount);
    NPar::TJobExecutor exec(&job, environment);
    TVector<typename TMapper::TOutput> mapperOutput;
    exec.GetResultVec(&mapperOutput);
    return mapperOutput;
}

template <typename TError, typename TApproxDefs>
void MapSetApproxes(const TSplitTree& splitTree, const TDatasetPtrs& testDataPtrs, TVector<TVector<double>>* averageLeafValues, TVector<double>* sumLeafWeights, TLearnContext* ctx) {
    static_assert(TError::IsCatboostErrorFunction, "TError is not a CatBoost error function class");

    using namespace NCatboostDistributed;
    using TSum = typename TApproxDefs::TSumType;
    using TPairwiseBuckets = typename TApproxDefs::TPairwiseBuckets;
    using TBucketUpdater = typename TApproxDefs::TBucketUpdater;
    using TDeltaUpdater = typename TApproxDefs::TDeltaUpdater;

    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    const int workerCount = ctx->RootEnvironment->GetSlaveCount();
    ApplyMapper<TCalcApproxStarter>(workerCount, ctx->SharedTrainData, MakeEnvelope(splitTree));
    const int gradientIterations = ctx->Params.ObliviousTreeOptions->LeavesEstimationIterations;
    const int approxDimension = ctx->LearnProgress.ApproxDimension;
    const int leafCount = splitTree.GetLeafCount();
    TVector<TSum> buckets(leafCount, TSum(gradientIterations, approxDimension, TError::GetHessianType()));
    averageLeafValues->resize(approxDimension, TVector<double>(leafCount));
    for (int it = 0; it < gradientIterations; ++it) {
        TPairwiseBuckets pairwiseBuckets;
        TApproxDefs::SetPairwiseBucketsSize(leafCount, &pairwiseBuckets);
        const auto bucketsFromAllWorkers = ApplyMapper<TBucketUpdater>(workerCount, ctx->SharedTrainData);
        // reduce across workers
        for (int workerIdx = 0; workerIdx < workerCount; ++workerIdx) {
            const auto& workerBuckets = bucketsFromAllWorkers[workerIdx].Data.first;
            for (int leafIdx = 0; leafIdx < leafCount; ++leafIdx) {
                if (ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Gradient) {
                    buckets[leafIdx].AddDerWeight(workerBuckets[leafIdx].SumDerHistory[it], workerBuckets[leafIdx].SumWeights, it);
                } else {
                    Y_ASSERT(ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod == ELeavesEstimation::Newton);
                    buckets[leafIdx].AddDerDer2(workerBuckets[leafIdx].SumDerHistory[it], workerBuckets[leafIdx].SumDer2History[it], it);
                }
            }
            TApproxDefs::AddPairwiseBuckets(bucketsFromAllWorkers[workerIdx].Data.second, &pairwiseBuckets);
        }
        const auto leafValues = TApproxDefs::CalcLeafValues(buckets, pairwiseBuckets, it, *ctx);
        AddElementwise(leafValues, averageLeafValues);
        // calc model and update approx deltas on workers
        ApplyMapper<TDeltaUpdater>(workerCount, ctx->SharedTrainData, leafValues);
    }

    const auto leafWeightsFromAllWorkers = ApplyMapper<TLeafWeightsGetter>(workerCount, ctx->SharedTrainData); // [workerIdx][dimIdx][leafIdx]
    sumLeafWeights->resize(leafCount);
    for (const auto& workerLeafWeights : leafWeightsFromAllWorkers) {
        AddElementwise(workerLeafWeights, sumLeafWeights);
    }

    NormalizeLeafValues(
        IsPairwiseError(ctx->Params.LossFunctionDescription->GetLossFunction()),
        ctx->Params.BoostingOptions->LearningRate,
        *sumLeafWeights,
        averageLeafValues
    );

    // update learn approx and average approx
    ApplyMapper<TApproxUpdater>(workerCount, ctx->SharedTrainData, *averageLeafValues);
    // update test
    const auto indices = BuildIndices(/*unused fold*/{}, splitTree, /*learnData*/ {}, testDataPtrs, &ctx->LocalExecutor);
    const bool storeExpApprox = IsStoreExpApprox(ctx->Params.LossFunctionDescription->GetLossFunction());
    UpdateAvrgApprox(storeExpApprox, /*learnSampleCount*/ 0, indices, *averageLeafValues, testDataPtrs, &ctx->LearnProgress, &ctx->LocalExecutor);
}

template <typename TError>
struct TSetApproxesSimpleDefs {
    using TSumType = TSum;
    using TPairwiseBuckets = TArray2D<double>;
    using TBucketUpdater = NCatboostDistributed::TBucketSimpleUpdater<TError>;
    using TDeltaUpdater = NCatboostDistributed::TDeltaSimpleUpdater;
    static void SetPairwiseBucketsSize(size_t leafCount, TPairwiseBuckets* pairwiseBuckets) {
        pairwiseBuckets->SetSizes(leafCount, leafCount);
        pairwiseBuckets->FillZero();
    }
    static void AddPairwiseBuckets(const TPairwiseBuckets& increment, TPairwiseBuckets* total) {
        Y_ASSERT(increment.GetXSize() == total->GetXSize() && increment.GetYSize() == total->GetYSize());
        for (size_t winnerIdx = 0; winnerIdx < increment.GetYSize(); ++winnerIdx) {
            for (size_t loserIdx = 0; loserIdx < increment.GetXSize(); ++loserIdx) {
                (*total)[winnerIdx][loserIdx] += increment[winnerIdx][loserIdx];
            }
        }
    }
    static TVector<TVector<double>> CalcLeafValues(const TVector<TSumType>& buckets,
        const TPairwiseBuckets& pairwiseBuckets,
        int iterationNum,
        const TLearnContext& ctx
    ) {
        const size_t leafCount = buckets.size();
        TVector<TVector<double>> leafValues(/*dimensionCount*/ 1, TVector<double>(leafCount));
        const size_t allDocCount = ctx.LearnProgress.Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress.Folds[0].GetSumWeight();
        CalcMixedModelSimple(buckets, pairwiseBuckets, iterationNum, ctx.Params, sumAllWeights, allDocCount, &leafValues[0]);
        return leafValues;
    }
};

template <typename TError>
struct TSetApproxesMultiDefs {
    using TSumType = TSumMulti;
    using TPairwiseBuckets = NCatboostDistributed::TUnusedInitializedParam;
    using TBucketUpdater = NCatboostDistributed::TBucketMultiUpdater<TError>;
    using TDeltaUpdater = NCatboostDistributed::TDeltaMultiUpdater;
    static void SetPairwiseBucketsSize(size_t /*leafCount*/, TPairwiseBuckets* /*pairwiseBuckets*/) {}
    static void AddPairwiseBuckets(const TPairwiseBuckets& /*increment*/, TPairwiseBuckets* /*total*/) {}
    static TVector<TVector<double>> CalcLeafValues(const TVector<TSumType>& buckets,
        const TPairwiseBuckets& /*pairwiseBuckets*/,
        int iterationNum,
        const TLearnContext& ctx
    ) {
        const int dimensionCount = ctx.LearnProgress.ApproxDimension;
        const size_t leafCount = buckets.size();
        TVector<TVector<double>> leafValues(dimensionCount, TVector<double>(leafCount));
        const auto estimationMethod = ctx.Params.ObliviousTreeOptions->LeavesEstimationMethod;
        const float l2Regularizer = ctx.Params.ObliviousTreeOptions->L2Reg;
        const size_t allDocCount = ctx.LearnProgress.Folds[0].GetLearnSampleCount();
        const double sumAllWeights = ctx.LearnProgress.Folds[0].GetSumWeight();
        if (estimationMethod == ELeavesEstimation::Newton) {
            CalcMixedModelMulti(CalcModelNewtonMulti, buckets, iterationNum, l2Regularizer, sumAllWeights, allDocCount, &leafValues);
        } else {
            Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
            CalcMixedModelMulti(CalcModelGradientMulti, buckets, iterationNum, l2Regularizer, sumAllWeights, allDocCount, &leafValues);
        }
        return leafValues;
    }
};

template <typename TError>
void MapSetApproxesSimple(const TSplitTree& splitTree, const TDatasetPtrs& testDataPtrs, TVector<TVector<double>>* averageLeafValues, TVector<double>* sumLeafWeights, TLearnContext* ctx) {
    MapSetApproxes<TError, TSetApproxesSimpleDefs<TError>>(splitTree, testDataPtrs, averageLeafValues, sumLeafWeights, ctx);
}

template <typename TError>
void MapSetApproxesMulti(const TSplitTree& splitTree, const TDatasetPtrs& testDataPtrs, TVector<TVector<double>>* averageLeafValues, TVector<double>* sumLeafWeights, TLearnContext* ctx) {
    MapSetApproxes<TError, TSetApproxesMultiDefs<TError>>(splitTree, testDataPtrs, averageLeafValues, sumLeafWeights, ctx);
}

template <typename TError>
void MapSetDerivatives(TLearnContext* ctx) {
    static_assert(TError::IsCatboostErrorFunction, "TError is not a CatBoost error function class");

    using namespace NCatboostDistributed;
    Y_ASSERT(ctx->Params.SystemOptions->IsMaster());
    ApplyMapper<TDerivativeSetter<TError>>(ctx->RootEnvironment->GetSlaveCount(), ctx->SharedTrainData);
}
