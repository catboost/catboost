#pragma once

#include "approx_calcer_helpers.h"
#include "approx_calcer_multi.h"
#include "approx_calcer_querywise.h"
#include "fold.h"
#include "score_calcer.h"
#include "index_calcer.h"
#include "learn_context.h"
#include "error_functions.h"

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
    TLearnContext* ctx,
    TVector<double>* leafValues,
    TVector<double>* resArr
) {
    ExpApproxIf(StoreExpApprox, leafValues);

    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues->data();

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, docCount);
    blockParams.SetBlockSize(1000);

    ctx->LocalExecutor.ExecRange([=] (int blockIdx) {
        UpdateApproxBlock<StoreExpApprox>(blockParams, leafValuesData, indicesData, blockIdx, resArrData);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

namespace {
constexpr int APPROX_BLOCK_SIZE = 500;

template <typename TError>
void CalcShiftedApproxDers(
    const TVector<double>& approxes,
    const TVector<double>& approxesDelta,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TError& error,
    int sampleStart,
    int sampleFinish,
    TVector<TDer1Der2>* weightedDers,
    TLearnContext* ctx
) {
    NPar::TLocalExecutor::TExecRangeParams blockParams(sampleStart, sampleFinish);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor.ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        error.CalcDersRange(
            blockOffset,
            Min(blockParams.GetBlockSize(), sampleFinish - blockOffset),
            approxes.data(),
            approxesDelta.data(),
            targets.data(),
            weights.data(),
            weightedDers->data() - sampleStart
        );
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
} // anonymous namespace

template <ELeavesEstimation LeafEstimationType, typename TError>
void CalcApproxDersRange(
    const TIndexType* indices,
    const float* targets,
    const float* weights,
    const double* approxes,
    const double* approxesDelta,
    const TError& error,
    int sampleCount,
    int iteration,
    TLearnContext* ctx,
    TVector<TSum>* buckets,
    TDer1Der2* weightedDers
) {
    const int leafCount = buckets->ysize();

    NPar::TLocalExecutor::TExecRangeParams blockParams(0, sampleCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    TVector<TVector<TDer1Der2>> blockBucketDers(blockParams.GetBlockCount(), TVector<TDer1Der2>(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0}));
    TVector<TDer1Der2>* blockBucketDersData = blockBucketDers.data();
    // TODO(espetrov): Do not calculate sumWeights for Newton.
    // TODO(espetrov): Calculate sumWeights only on first iteration for Gradient, because on next iteration it is the same.
    // Check speedup on flights dataset.
    TVector<TVector<double>> blockBucketSumWeights(blockParams.GetBlockCount(), TVector<double>(leafCount, 0));
    TVector<double>* blockBucketSumWeightsData = blockBucketSumWeights.data();

    ctx->LocalExecutor.ExecRange([=](int blockId) {
        constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
        TDer1Der2* approxesDer = weightedDers + innerBlockSize * blockId;

        const int blockStart = blockId * blockParams.GetBlockSize();
        const int nextBlockStart = Min(sampleCount, blockStart + blockParams.GetBlockSize());

        TDer1Der2* bucketDers = blockBucketDersData[blockId].data();
        double* bucketSumWeights = blockBucketSumWeightsData[blockId].data();

        for (int innerBlockStart = blockStart; innerBlockStart < nextBlockStart; innerBlockStart += innerBlockSize) {
            const int nextInnerBlockStart = Min(nextBlockStart, innerBlockStart + innerBlockSize);
            error.CalcDersRange(
                innerBlockStart,
                nextInnerBlockStart - innerBlockStart,
                approxes,
                approxesDelta,
                targets,
                weights,
                approxesDer - innerBlockStart
            );
            if (weights != nullptr) {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDer1Der2& ders = bucketDers[indices[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indices[z]] += weights[z];
                }
            } else {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDer1Der2& ders = bucketDers[indices[z]];
                    ders.Der1 += approxesDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxesDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indices[z]] += 1;
                }
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
            if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                UpdateBucket<LeafEstimationType>(
                    blockBucketDers[blockId][leafId],
                    blockBucketSumWeights[blockId][leafId],
                    iteration,
                    &(*buckets)[leafId]
                );
            }
        }
    }
}

template <ELeavesEstimation LeafEstimationType, typename TError>
void CalcApproxDeltaIterationSimple(
    const TVector<TIndexType>& indices,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    const TFold::TBodyTail& bt,
    const TError& error,
    int iteration,
    float l2Regularizer,
    TLearnContext* ctx,
    TVector<TSum>* buckets,
    TVector<double>* resArr,
    TVector<TDer1Der2>* weightedDers
) {
    int leafCount = buckets->ysize();

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange<LeafEstimationType>(
            indices.data(),
            targets.data(),
            weights.data(),
            bt.Approx[0].data(),
            resArr->data(),
            error,
            bt.BodyFinish,
            iteration,
            ctx,
            buckets,
            weightedDers->data()
        );
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        Y_ASSERT(LeafEstimationType == ELeavesEstimation::Gradient);
        CalculateDersForQueries(
            bt.Approx[0],
            *resArr,
            targets,
            weights,
            queriesInfo,
            error,
            /*queryStartIndex=*/0,
            bt.BodyQueryFinish,
            weightedDers
        );
        UpdateBucketsForQueries(
            *weightedDers,
            indices,
            weights,
            queriesInfo,
            /*queryStartIndex=*/0,
            bt.BodyQueryFinish,
            iteration,
            buckets
        );
    }

    // compute mixed model
    TVector<double> curLeafValues(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    // compute tail
    if (!ctx->Params.BoostingOptions->ApproxOnFullHistory) {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.TailFinish, ctx, &curLeafValues, resArr);
    } else {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.BodyFinish, ctx, &curLeafValues, resArr);

        if (error.GetErrorType() == EErrorType::PerObjectError) {
            CalcShiftedApproxDers(bt.Approx[0], *resArr, targets, weights, error, bt.BodyFinish, bt.TailFinish, weightedDers, ctx);
        } else {
            Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
            CalculateDersForQueries(
                bt.Approx[0],
                *resArr,
                targets,
                weights,
                queriesInfo,
                error,
                bt.BodyQueryFinish,
                bt.TailQueryFinish,
                weightedDers
            );
        }

        TSum* bucketsData = buckets->data();
        const TIndexType* indicesData = indices.data();
        const TDer1Der2* scratchDersData = weightedDers->data();
        double* resArrData = resArr->data();
        TVector<double> avrg;
        avrg.yresize(1);
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weights.empty() ? 1 : weights[z];
            UpdateBucket<LeafEstimationType>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<LeafEstimationType>(bucket, iteration, l2Regularizer);
            ExpApproxIf(TError::StoreExpApprox, &avrg);
            resArrData[z] = UpdateApprox<TError::StoreExpApprox>(resArrData[z], avrg[0]);
        }
    }
}

template <typename TError>
void CalcApproxDeltaSimple(
    const TFold& ff,
    const TSplitTree& tree,
    const TError& error,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta,
    TVector<TIndexType>* ind
) {
    auto& indices = *ind;
    approxesDelta->resize(ff.BodyTailArr.ysize());

    TVector<float> pairwiseWeights;
    if (error.GetErrorType() == EErrorType::PairwiseError) {
        pairwiseWeights = CalcPairwiseWeights(ff.LearnQueriesInfo);
    }
    const TVector<float>& weights = error.GetErrorType() == EErrorType::PairwiseError ? pairwiseWeights : ff.LearnWeights;

    ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
        const TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];

        TVector<TVector<double>>& resArr = (*approxesDelta)[bodyTailId];
        const double initValue = GetNeutralApprox<TError::StoreExpApprox>();
        if (resArr.empty()) {
            resArr.resize(1);
            resArr[0].yresize(bt.TailFinish);
        }
        Fill(resArr[0].begin(), resArr[0].end(), initValue);

        const int leafCount = tree.GetLeafCount();
        const int scratchSize = Max(
            !ctx->Params.BoostingOptions->ApproxOnFullHistory ? 0 : bt.TailFinish - bt.BodyFinish,
            error.GetErrorType() == EErrorType::PerObjectError ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT : bt.BodyFinish
        );
        TVector<TDer1Der2> weightedDers;
        weightedDers.yresize(scratchSize);
        const auto treeLearnerOptions = ctx->Params.ObliviousTreeOptions.Get();
        const int gradientIterations = static_cast<int>(treeLearnerOptions.LeavesEstimationIterations);
        TVector<TSum> buckets(leafCount, TSum(gradientIterations));

        const ELeavesEstimation estimationMethod = treeLearnerOptions.LeavesEstimationMethod;
        const float l2Regularizer = treeLearnerOptions.L2Reg;
        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeavesEstimation::Newton) {
                CalcApproxDeltaIterationSimple<ELeavesEstimation::Newton>(
                    indices,
                    ff.LearnTarget,
                    weights,
                    ff.LearnQueriesInfo,
                    bt,
                    error,
                    it,
                    l2Regularizer,
                    ctx,
                    &buckets,
                    &resArr[0],
                    &weightedDers
                );
            } else {
                CB_ENSURE(estimationMethod == ELeavesEstimation::Gradient);
                CalcApproxDeltaIterationSimple<ELeavesEstimation::Gradient>(
                    indices,
                    ff.LearnTarget,
                    weights,
                    ff.LearnQueriesInfo,
                    bt,
                    error,
                    it,
                    l2Regularizer,
                    ctx,
                    &buckets,
                    &resArr[0],
                    &weightedDers
                );
            }
        }
    }, 0, ff.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <typename TError>
void CalcApproxDelta(
    const TFold& ff,
    const TSplitTree& tree,
    const TError& error,
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta,
    TVector<TIndexType>* ind
) {
    int approxDimension = ff.GetApproxDimension();
    if (approxDimension == 1) {
        CalcApproxDeltaSimple(ff, tree, error, ctx, approxesDelta, ind);
    } else {
        CalcApproxDeltaMulti(ff, tree, error, ctx, approxesDelta, ind);
    }
}

template <ELeavesEstimation LeafEstimationType, typename TError>
void CalcLeafValuesIterationSimple(
    const TVector<TIndexType>& indices,
    const TVector<float>& targets,
    const TVector<float>& weights,
    const TVector<TQueryInfo>& queriesInfo,
    const TError& error,
    int iteration,
    float l2Regularizer,
    TLearnContext* ctx,
    TVector<TSum>* buckets,
    TVector<double>* approxes,
    TVector<TDer1Der2>* weightedDers
) {
    int leafCount = buckets->ysize();
    int learnSampleCount = approxes->ysize();

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange<LeafEstimationType>(
            indices.data(),
            targets.data(),
            weights.data(),
            approxes->data(),
            /*resArr=*/nullptr,
            error,
            learnSampleCount,
            iteration,
            ctx,
            buckets,
            weightedDers->data()
        );
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError || error.GetErrorType() == EErrorType::PairwiseError);
        Y_ASSERT(LeafEstimationType == ELeavesEstimation::Gradient);
        CalculateDersForQueries(
            *approxes,
            /*approxesDelta=*/{},
            targets,
            weights,
            queriesInfo,
            error,
            /*queryStartIndex=*/0,
            queriesInfo.ysize(),
            weightedDers
        );
        UpdateBucketsForQueries(
            *weightedDers,
            indices,
            weights,
            queriesInfo,
            /*queryStartIndex=*/0,
            queriesInfo.ysize(),
            iteration,
            buckets
        );
    }

    TVector<double> curLeafValues;
    curLeafValues.yresize(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    UpdateApproxDeltas<TError::StoreExpApprox>(indices, learnSampleCount, ctx, &curLeafValues, approxes);
}

template <typename TError>
void CalcLeafValuesSimple(
    const TDataset& learnData,
    const TDataset* testData,
    const TSplitTree& tree,
    const TError& error,
    const TFold& ff,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues,
    TVector<TIndexType>* ind
) {
    auto& indices = *ind;
    indices = BuildIndices(ff, tree, learnData, testData, &ctx->LocalExecutor);

    const TFold::TBodyTail& bt = ff.BodyTailArr[0];
    const int leafCount = tree.GetLeafCount();
    const int learnSampleCount = learnData.GetSampleCount();

    TVector<TVector<double>> approxes(1);
    approxes[0].assign(bt.Approx[0].begin(), bt.Approx[0].begin() + learnSampleCount);

    const auto& learnerOptions = ctx->Params.ObliviousTreeOptions.Get();
    const int gradientIterations = learnerOptions.LeavesEstimationIterations;
    TVector<TSum> buckets(leafCount, gradientIterations);
    const int scratchSize = error.GetErrorType() == EErrorType::PerObjectError
        ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT
        : learnSampleCount;
    TVector<TDer1Der2> weightedDers(scratchSize);

    TVector<float> pairwiseWeights;
    if (error.GetErrorType() == EErrorType::PairwiseError) {
        pairwiseWeights = CalcPairwiseWeights(ff.LearnQueriesInfo);
    }
    const TVector<float>& weights = error.GetErrorType() == EErrorType::PairwiseError ? pairwiseWeights : ff.LearnWeights;

    const ELeavesEstimation estimationMethod = learnerOptions.LeavesEstimationMethod;
    const float l2Regularizer = learnerOptions.L2Reg;
    for (int it = 0; it < gradientIterations; ++it) {
        if (estimationMethod == ELeavesEstimation::Newton) {
            CalcLeafValuesIterationSimple<ELeavesEstimation::Newton>(
                indices,
                ff.LearnTarget,
                weights,
                ff.LearnQueriesInfo,
                error,
                it,
                l2Regularizer,
                ctx,
                &buckets,
                &approxes[0],
                &weightedDers
            );
        } else {
            CB_ENSURE(estimationMethod == ELeavesEstimation::Gradient);
            CalcLeafValuesIterationSimple<ELeavesEstimation::Gradient>(
                indices,
                ff.LearnTarget,
                weights,
                ff.LearnQueriesInfo,
                error,
                it,
                l2Regularizer,
                ctx,
                &buckets,
                &approxes[0],
                &weightedDers
            );
        }
    }

    leafValues->assign(1, TVector<double>(leafCount));
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
    TVector<TIndexType>* ind
) {
    const int approxDimension = ctx->LearnProgress.AveragingFold.GetApproxDimension();
    if (approxDimension == 1) {
        CalcLeafValuesSimple(learnData, testData, tree, error, fold, ctx, leafValues, ind);
    } else {
        CalcLeafValuesMulti(learnData, testData, tree, error, fold, ctx, leafValues, ind);
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
    TLearnContext* ctx,
    TVector<TVector<TVector<double>>>* approxesDelta// [bodyTailId][approxDim][docIdxInPermuted]
) {
    TVector<TIndexType> indices = BuildIndices(fold, tree, learnData, testData, &ctx->LocalExecutor);
    CalcApproxDelta(fold, tree, error, ctx, approxesDelta, &indices);
}
