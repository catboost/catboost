#pragma once

#include "approx_calcer_helpers.h"
#include "approx_calcer_multi.h"
#include "approx_calcer_querywise.h"
#include "fold.h"
#include "score_calcer.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "error_functions.h"
#include "yetirank_helpers.h"

#include <catboost/libs/logging/logging.h>
#include <catboost/libs/logging/profile_info.h>

template <bool StoreExpApprox, int VectorWidth>
inline void UpdateApproxKernel(const double* leafValues, const TIndexType* indices, double* resArr) {
    Y_ASSERT(VectorWidth == 4);
    const TIndexType idx0 = indices[0];
    const TIndexType idx1 = indices[1];
    const TIndexType idx2 = indices[2];
    const TIndexType idx3 = indices[3];
    const double resArr0 = resArr[0];
    const double resArr1 = resArr[1];
    const double resArr2 = resArr[2];
    const double resArr3 = resArr[3];
    const double value0 = leafValues[idx0];
    const double value1 = leafValues[idx1];
    const double value2 = leafValues[idx2];
    const double value3 = leafValues[idx3];
    resArr[0] = UpdateApprox<StoreExpApprox>(resArr0, value0);
    resArr[1] = UpdateApprox<StoreExpApprox>(resArr1, value1);
    resArr[2] = UpdateApprox<StoreExpApprox>(resArr2, value2);
    resArr[3] = UpdateApprox<StoreExpApprox>(resArr3, value3);
}

template <bool StoreExpApprox>
inline void UpdateApproxBlock(
    const NPar::TLocalExecutor::TExecRangeParams& params,
    const double* leafValues,
    const TIndexType* indices,
    int blockIdx,
    double* resArr
) {
    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int VectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + VectorWidth <= nextBlockStart; doc += VectorWidth) {
        UpdateApproxKernel<StoreExpApprox, VectorWidth>(leafValues, indices + doc, resArr + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        resArr[doc] = UpdateApprox<StoreExpApprox>(resArr[doc], leafValues[indices[doc]]);
    }
}

template <bool StoreExpApprox>
inline void UpdateApproxDeltas(
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<double>* leafValues,
    TVector<double>* resArr
) {
    ExpApproxIf(StoreExpApprox, leafValues);

    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues->data();

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(1000);

    localExecutor->ExecRange([=] (int blockIdx) {
        UpdateApproxBlock<StoreExpApprox>(blockParams, leafValuesData, indicesData, blockIdx, resArrData);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

namespace {
static constexpr int APPROX_BLOCK_SIZE = 500;

template <typename TError>
void CalcShiftedApproxDers(
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TError& error,
    int sampleStart,
    int sampleFinish,
    TVector<TDers>* weightedDers,
    TLearnContext* ctx
) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(sampleStart, sampleFinish);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor.ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        error.CalcDersRange(
            blockOffset,
            Min(blockParams.GetBlockSize(), sampleFinish - blockOffset),
            /*calcThirdDer=*/false,
            approxes.data(),
            approxesDelta.data(),
            targets.data(),
            weights.data(),
            weightedDers->data() - sampleStart
        );
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
} // anonymous namespace

template <typename TError>
void CalcApproxDersRange(
    const TVector<TIndexType>& indices,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TError& error,
    int sampleCount,
    int iteration,
    ELeavesEstimation estimationMethod,
    NPar::TLocalExecutor* localExecutor,
    TVector<TSum>* buckets,
    TVector<TDers>* weightedDers
) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(0, sampleCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    const int leafCount = buckets->ysize();
    TVector<TVector<TDers>> blockBucketDers(blockParams.GetBlockCount(), TVector<TDers>(leafCount, TDers{/*Der1*/0.0, /*Der2*/0.0, /*Der3*/0.0}));
    TVector<TDers>* blockBucketDersData = blockBucketDers.data();
    // TODO(espetrov): Do not calculate sumWeights for Newton.
    // TODO(espetrov): Calculate sumWeights only on first iteration for Gradient, because on next iteration it is the same.
    // Check speedup on flights dataset.
    TVector<TVector<double>> blockBucketSumWeights(blockParams.GetBlockCount(), TVector<double>(leafCount, 0));
    TVector<double>* blockBucketSumWeightsData = blockBucketSumWeights.data();
    const TIndexType* indicesData = indices.data();
    const float* targetsData = targets.data();
    const float* weightsData = weights.data();
    const double* approxesData = approxes.data();
    const double* approxesDeltaData = approxesDelta.data();
    TDers* weightedDersData = weightedDers->data();
    localExecutor->ExecRange([=](int blockId) {
        constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
        TDers* approxesDer = weightedDersData + innerBlockSize * blockId;

        const int blockStart = blockId * blockParams.GetBlockSize();
        const int nextBlockStart = Min(sampleCount, blockStart + blockParams.GetBlockSize());

        TDers* bucketDers = blockBucketDersData[blockId].data();
        double* bucketSumWeights = blockBucketSumWeightsData[blockId].data();

        for (int innerBlockStart = blockStart; innerBlockStart < nextBlockStart; innerBlockStart += innerBlockSize) {
            const int nextInnerBlockStart = Min(nextBlockStart, innerBlockStart + innerBlockSize);
            error.CalcDersRange(
                innerBlockStart,
                nextInnerBlockStart - innerBlockStart,
                /*calcThirdDer=*/false,
                approxesData,
                approxesDeltaData,
                targetsData,
                weightsData,
                approxesDer - innerBlockStart
            );
            if (weightsData != nullptr) {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDers& ders = bucketDers[indicesData[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indicesData[z]] += weightsData[z];
                }
            } else {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDers& ders = bucketDers[indicesData[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indicesData[z]] += 1;
                }
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leafId = 0; leafId < leafCount; ++leafId) {
            for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
                if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                    UpdateBucket<ELeavesEstimation::Newton>(
                        blockBucketDers[blockId][leafId],
                        blockBucketSumWeights[blockId][leafId],
                        iteration,
                        &(*buckets)[leafId]
                    );
                }
            }
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leafId = 0; leafId < leafCount; ++leafId) {
            for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
                if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                    UpdateBucket<ELeavesEstimation::Gradient>(
                        blockBucketDers[blockId][leafId],
                        blockBucketSumWeights[blockId][leafId],
                        iteration,
                        &(*buckets)[leafId]
                    );
                }
            }
        }
    }
}

template <typename TError>
void UpdateBucketsSimple(
    const TVector<TIndexType>& indices,
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const TVector<double>& approxes,
    const TVector<double>& approxDeltas,
    const TError& error,
    int sampleCount,
    int queryCount,
    int iteration,
    ELeavesEstimation estimationMethod,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TSum>* buckets,
    TVector<TDers>* scratchDers
) {
    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange(
            indices,
            ff.LearnTarget,
            ff.LearnWeights,
            approxes,
            approxDeltas,
            error,
            sampleCount,
            iteration,
            estimationMethod,
            localExecutor,
            buckets,
            scratchDers
        );
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);

        TVector<TQueryInfo> recalculatedQueriesInfo;
        TVector<float> recalculatedPairwiseWeights;
        const bool isYetiRank = params.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRank;
        if (isYetiRank) {
            YetiRankRecalculation(ff, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &recalculatedPairwiseWeights);
        }
        const TVector<TQueryInfo>& queriesInfo = isYetiRank ? recalculatedQueriesInfo : ff.LearnQueriesInfo;
        const TVector<float>& weights = bt.PairwiseWeights.empty() ? ff.LearnWeights : isYetiRank ? recalculatedPairwiseWeights : bt.PairwiseWeights;

        CalculateDersForQueries(
            approxes,
            approxDeltas,
            ff.LearnTarget,
            weights,
            queriesInfo,
            error,
            /*queryStartIndex=*/0,
            queryCount,
            scratchDers
        );
        UpdateBucketsForQueries(
            *scratchDers,
            indices,
            weights,
            queriesInfo,
            /*queryStartIndex=*/0,
            queryCount,
            iteration,
            buckets
        );
    }
}

inline void CalcMixedModelSimple(const TVector<TSum>& buckets,
    int iteration,
    const NCatboostOptions::TCatBoostOptions& params,
    TVector<double>* leafValues
) {
    const int leafCount = buckets.ysize();
    leafValues->yresize(leafCount);
    const ELeavesEstimation estimationMethod = params.ObliviousTreeOptions->LeavesEstimationMethod;
    const float l2Regularizer = params.ObliviousTreeOptions->L2Reg;
    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            (*leafValues)[leaf] = CalcModel<ELeavesEstimation::Newton>(buckets[leaf], iteration, l2Regularizer);
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            (*leafValues)[leaf] = CalcModel<ELeavesEstimation::Gradient>(buckets[leaf], iteration, l2Regularizer);
        }
    }
}

template <typename TError>
void CalcTailModelSimple(
    const TVector<TIndexType>& indices,
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const TError& error,
    int iteration,
    float l2Regularizer,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TLearnContext* ctx,
    TVector<TSum>* buckets,
    TVector<double>* approxDeltas,
    TVector<TDers>* weightedDers
) {
    TVector<TQueryInfo> recalculatedQueriesInfo;
    TVector<float> recalculatedPairwiseWeights;
    const bool isYetiRank = params.LossFunctionDescription->GetLossFunction() == ELossFunction::YetiRank;
    if (isYetiRank) {
        YetiRankRecalculation(ff, bt, params, randomSeed, localExecutor, &recalculatedQueriesInfo, &recalculatedPairwiseWeights);
    }
    const TVector<TQueryInfo>& queriesInfo = isYetiRank ? recalculatedQueriesInfo : ff.LearnQueriesInfo;
    const TVector<float>& weights = bt.PairwiseWeights.empty() ? ff.LearnWeights : isYetiRank ? recalculatedPairwiseWeights : bt.PairwiseWeights;

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcShiftedApproxDers(bt.Approx[0], *approxDeltas, ff.LearnTarget, weights, error, bt.BodyFinish, bt.TailFinish, weightedDers, ctx);
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        CalculateDersForQueries(bt.Approx[0], *approxDeltas, ff.LearnTarget, weights, queriesInfo, error, bt.BodyQueryFinish, bt.TailQueryFinish, weightedDers);
    }
    TSum* bucketsData = buckets->data();
    const TIndexType* indicesData = indices.data();
    const TDers* scratchDersData = weightedDers->data();
    double* approxDeltasData = approxDeltas->data();
    TVector<double> avrg;
    avrg.yresize(1);
    const auto treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
    if (estimationMethod == ELeavesEstimation::Newton) {
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weights.empty() ? 1 : weights[z];
            UpdateBucket<ELeavesEstimation::Newton>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<ELeavesEstimation::Newton>(bucket, iteration, l2Regularizer);
            ExpApproxIf(TError::StoreExpApprox, &avrg);
            approxDeltasData[z] = UpdateApprox<TError::StoreExpApprox>(approxDeltasData[z], avrg[0]);
        }
    } else {
        Y_ASSERT(estimationMethod == ELeavesEstimation::Gradient);
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weights.empty() ? 1 : weights[z];
            UpdateBucket<ELeavesEstimation::Gradient>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<ELeavesEstimation::Gradient>(bucket, iteration, l2Regularizer);
            ExpApproxIf(TError::StoreExpApprox, &avrg);
            approxDeltasData[z] = UpdateApprox<TError::StoreExpApprox>(approxDeltasData[z], avrg[0]);
        }
    }
}

template <typename TError>
void CalcApproxDeltaSimple(
    const TFold& ff,
    int leafCount,
    const TError& error,
    const TVector<TIndexType>& indices,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta
) {
    const auto treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = static_cast<int>(treeLearnerOptions.LeavesEstimationIterations);
    const float l2Regularizer = treeLearnerOptions.L2Reg;
    approxesDelta->resize(ff.BodyTailArr.ysize());
    const TVector<ui64> randomSeeds = GenRandUI64Vector(ff.BodyTailArr.ysize(), randomSeed);
    ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
        const TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];

        TVector<TVector<double>>& resArr = (*approxesDelta)[bodyTailId];
        const double initValue = GetNeutralApprox<TError::StoreExpApprox>();
        if (resArr.empty()) {
            resArr.resize(1);
            resArr[0].yresize(bt.TailFinish);
        }
        Fill(resArr[0].begin(), resArr[0].end(), initValue);

        const int scratchSize = Max(
            !ctx->Params.BoostingOptions->ApproxOnFullHistory ? 0 : bt.TailFinish - bt.BodyFinish,
            error.GetErrorType() == EErrorType::PerObjectError ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT : bt.BodyFinish
        );
        const auto estimationMethod = ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod;
        auto& localExecutor = ctx->LocalExecutor;

        TVector<TDers> weightedDers;
        weightedDers.yresize(scratchSize); // thread scratch
        TVector<TSum> buckets(leafCount, TSum(gradientIterations)); // thread scratch
        TVector<double> curLeafValues; // thread scratch
        for (int it = 0; it < gradientIterations; ++it) {
            UpdateBucketsSimple(indices, ff, bt, bt.Approx[0], resArr[0], error, bt.BodyFinish, bt.BodyQueryFinish, it, estimationMethod, ctx->Params, randomSeeds[bodyTailId], &localExecutor, &buckets, &weightedDers);
            CalcMixedModelSimple(buckets, it, ctx->Params, &curLeafValues);
            if (!ctx->Params.BoostingOptions->ApproxOnFullHistory) {
                UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.TailFinish, &ctx->LocalExecutor, &curLeafValues, &resArr[0]);
            } else {
                UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.BodyFinish, &ctx->LocalExecutor, &curLeafValues, &resArr[0]);
                CalcTailModelSimple(indices, ff, bt, error, it, l2Regularizer, ctx->Params, randomSeeds[bodyTailId], &localExecutor, ctx, &buckets, &resArr[0], &weightedDers);
            }
        }
    }, 0, ff.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <typename TError>
void CalcApproxDelta(
    const TFold& ff,
    int leafCount,
    const TError& error,
    const TVector<TIndexType>& indices,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta
) {
    const int approxDimension = ff.GetApproxDimension();
    if (approxDimension == 1) {
        CalcApproxDeltaSimple(ff, leafCount, error, indices, randomSeed, ctx, approxesDelta);
    } else {
        CalcApproxDeltaMulti(ff, leafCount, error, indices, ctx, approxesDelta);
    }
}

template <typename TError>
void CalcLeafValuesSimple(
    int learnSampleCount,
    int leafCount,
    const TError& error,
    const TFold& ff,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues
) {
    const int scratchSize = error.GetErrorType() == EErrorType::PerObjectError
        ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT
        : learnSampleCount;
    TVector<TDers> weightedDers(scratchSize);

    const int queryCount = ff.LearnQueriesInfo.ysize();
    const auto& learnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = learnerOptions.LeavesEstimationIterations;
    const auto estimationMethod = ctx->Params.ObliviousTreeOptions->LeavesEstimationMethod;
    auto& localExecutor = ctx->LocalExecutor;

    const TFold::TBodyTail& bt = ff.BodyTailArr[0];
    TVector<double> approxes(bt.Approx[0].begin(), bt.Approx[0].begin() + learnSampleCount); // scratch
    TVector<TSum> buckets(leafCount, gradientIterations); // scratch
    TVector<double> curLeafValues; // scratch vector
    for (int it = 0; it < gradientIterations; ++it) {
        UpdateBucketsSimple(indices, ff, bt, approxes, /*approxDeltas*/ {}, error, learnSampleCount, queryCount, it, estimationMethod, ctx->Params, ctx->Rand.GenRand(), &localExecutor, &buckets, &weightedDers);
        CalcMixedModelSimple(buckets, it, ctx->Params, &curLeafValues);
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, learnSampleCount, &ctx->LocalExecutor, &curLeafValues, &approxes);
    }

    leafValues->assign(1, TVector<double>(leafCount));
    const float l2Regularizer = learnerOptions.L2Reg;
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        for (int it = 0; it < gradientIterations; ++it) {
            (*leafValues)[0][leaf] += (estimationMethod == ELeavesEstimation::Newton)
                ? CalcModelNewton(buckets[leaf], it, l2Regularizer)
                : CalcModelGradient(buckets[leaf], it, l2Regularizer);
        }
    }
}

template <typename TError>
void CalcLeafValues(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TFold& fold,
    const TSplitTree& tree,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues,
    TVector<TIndexType>* indices
) {
    *indices = BuildIndices(fold, tree, learnData, testData, &ctx->LocalExecutor);
    const int approxDimension = ctx->LearnProgress.AveragingFold.GetApproxDimension();
    const int learnSampleCount = learnData.GetSampleCount();
    const int leafCount = tree.GetLeafCount();
    if (approxDimension == 1) {
        CalcLeafValuesSimple(learnSampleCount, leafCount, error, fold, *indices, ctx, leafValues);
    } else {
        CalcLeafValuesMulti(learnSampleCount, leafCount, error, fold, *indices, ctx, leafValues);
    }
}

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
template <typename TError>
void CalcApproxForLeafStruct(
    const TDataset& learnData,
    const TDataset* testData,
    const TError& error,
    const TFold& fold,
    const TSplitTree& tree,
    ui64 randomSeed,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta // [bodyTailId][approxDim][docIdxInPermuted]
) {
    const TVector<TIndexType> indices = BuildIndices(fold, tree, learnData, testData, &ctx->LocalExecutor);
    const int leafCount = tree.GetLeafCount();
    CalcApproxDelta(fold, leafCount, error, indices, randomSeed, ctx, approxesDelta);
}
