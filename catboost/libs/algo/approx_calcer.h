#pragma once

#include "approx_calcer_helpers.h"
#include "approx_calcer_multi.h"
#include "approx_calcer_pairwise.h"
#include "approx_calcer_querywise.h"
#include "params.h"
#include "fold.h"
#include "bin_tracker.h"
#include "error_functions.h"
#include "index_hash_calcer.h"
#include "score_calcer.h"
#include "index_calcer.h"
#include "learn_context.h"
#include <catboost/libs/model/tensor_struct.h>

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
inline void UpdateApproxBlock(const NPar::TLocalExecutor::TBlockParams& params, const double* leafValues, const TIndexType* indices, int blockIdx, double* resArr) {
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
inline void UpdateApproxDeltas(const yvector<TIndexType>& indices,
                               int docCount,
                               TLearnContext* ctx,
                               yvector<double>* leafValues,
                               yvector<double>* resArr) {
    ExpApproxIf(StoreExpApprox, leafValues);

    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues->data();

    NPar::TLocalExecutor::TBlockParams blockParams(0, docCount);
    blockParams.SetBlockSize(10000).WaitCompletion();

    ctx->LocalExecutor.ExecRange([=] (int blockIdx) {
        UpdateApproxBlock<StoreExpApprox>(blockParams, leafValuesData, indicesData, blockIdx, resArrData);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

namespace {
constexpr int APPROX_BLOCK_SIZE = 500;

template <typename TError>
void CalcShiftedApproxDers(const yvector<double>& approx,
                           const yvector<double>& approxDelta,
                           const yvector<float>& target,
                           const yvector<float>& weight,
                           const TError& error,
                           int sampleStart,
                           int sampleFinish,
                           yvector<TDer1Der2>* scratchDers,
                           TLearnContext* ctx) {
    NPar::TLocalExecutor::TBlockParams blockParams(sampleStart, sampleFinish);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor.ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        error.CalcDersRange(blockOffset, Min(blockParams.GetBlockSize(), sampleFinish - blockOffset),
                            approx.data(),
                            approxDelta.data(),
                            target.data(),
                            weight.data(),
                            scratchDers->data() - sampleStart);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
} // anonymous namespace

template <ELeafEstimation LeafEstimationType, typename TError>
void CalcApproxDersRange(const TIndexType* indices,
                         const float* target,
                         const float* weight,
                         const double* approx,
                         const double* approxDeltas,
                         const TError& error,
                         int sampleCount,
                         int iteration,
                         TLearnContext* ctx,
                         yvector<TSum>* buckets,
                         TDer1Der2* scratchDers) {
    const int leafCount = buckets->ysize();

    NPar::TLocalExecutor::TBlockParams blockParams(0, sampleCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    yvector<yvector<TDer1Der2>> blockBucketDers(blockParams.GetBlockCount(), yvector<TDer1Der2>(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0}));
    yvector<TDer1Der2>* blockBucketDersData = blockBucketDers.data();
    yvector<yvector<double>> blockBucketSumWeights(blockParams.GetBlockCount(), yvector<double>(leafCount, 0));
    yvector<double>* blockBucketSumWeightsData = blockBucketSumWeights.data();

    ctx->LocalExecutor.ExecRange([=](int blockId) {
        constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
        TDer1Der2* approxDer = scratchDers + innerBlockSize * blockId;

        const int blockStart = blockId * blockParams.GetBlockSize();
        const int nextBlockStart = Min(sampleCount, blockStart + blockParams.GetBlockSize());

        TDer1Der2* bucketDers = blockBucketDersData[blockId].data();
        double* bucketSumWeights = blockBucketSumWeightsData[blockId].data();

        for (int innerBlockStart = blockStart; innerBlockStart < nextBlockStart; innerBlockStart += innerBlockSize) {
            const int nextInnerBlockStart = Min(nextBlockStart, innerBlockStart + innerBlockSize);
            error.CalcDersRange(innerBlockStart, nextInnerBlockStart - innerBlockStart,
                                approx,
                                approxDeltas,
                                target,
                                weight,
                                approxDer - innerBlockStart);
            if (weight != nullptr) {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDer1Der2& ders = bucketDers[indices[z]];
                    ders.Der1 += approxDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indices[z]] += weight[z];
                }
            } else {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    TDer1Der2& ders = bucketDers[indices[z]];
                    ders.Der1 += approxDer[z - innerBlockStart].Der1;
                    ders.Der2 += approxDer[z - innerBlockStart].Der2;
                    bucketSumWeights[indices[z]] += 1;
                }
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
            if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                UpdateBucket<LeafEstimationType>(blockBucketDers[blockId][leafId], blockBucketSumWeights[blockId][leafId], iteration, &(*buckets)[leafId]);
            }
        }
    }
}

template <ELeafEstimation LeafEstimationType, typename TError>
void CalcApproxDeltaIterationSimple(const yvector<TIndexType>& indices,
                                    const yvector<float>& target,
                                    const yvector<float>& weight,
                                    const yvector<ui32>& queriesId,
                                    const yhash<ui32, ui32>& queriesSize,
                                    const TFold::TBodyTail& bt,
                                    const TError& error,
                                    int iteration,
                                    float l2Regularizer,
                                    TLearnContext* ctx,
                                    yvector<TSum>* buckets,
                                    yvector<double>* resArr,
                                    yvector<TDer1Der2>* scratchDers) {
    int leafCount = buckets->ysize();

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange<LeafEstimationType>(indices.data(), target.data(), weight.data(), bt.Approx[0].data(), resArr->data(),
                                                error, bt.BodyFinish, iteration, ctx, buckets, scratchDers->data());
    } else if (error.GetErrorType() == EErrorType::PairwiseError) {
        Y_ASSERT(LeafEstimationType == ELeafEstimation::Gradient);
        CalcApproxDersRangePairs(indices, bt.Approx[0], *resArr, bt.Competitors, error,
                                 bt.BodyFinish, bt.TailFinish, iteration, buckets, scratchDers);
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError);
        Y_ASSERT(LeafEstimationType == ELeafEstimation::Gradient);
        CalcApproxDersQueriesRange(indices, bt.Approx[0], *resArr, target, weight, queriesId, queriesSize, error,
                                   bt.BodyFinish, bt.TailFinish, iteration, buckets, scratchDers);
    }

    // compute mixed model
    yvector<double> curLeafValues(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    // compute tail
    if (!ctx->Params.ApproxOnFullHistory) {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.TailFinish, ctx, &curLeafValues, resArr);
    } else {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.BodyFinish, ctx, &curLeafValues, resArr);

        if (error.GetErrorType() == EErrorType::PerObjectError) {
            CalcShiftedApproxDers(bt.Approx[0], *resArr, target, weight, error, bt.BodyFinish, bt.TailFinish, scratchDers, ctx);
        } else if (error.GetErrorType() == EErrorType::PairwiseError) {
            CalcShiftedApproxDersPairs(bt.Approx[0], *resArr, bt.Competitors, error, bt.BodyFinish, bt.TailFinish, scratchDers);
        } else {
            Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError);
            CalcShiftedApproxDersQueries(bt.Approx[0], *resArr, target, weight, queriesId, queriesSize, error, bt.BodyFinish, bt.TailFinish, scratchDers);
        }

        TSum* bucketsData = buckets->data();
        const TIndexType* indicesData = indices.data();
        const float* weightData = weight.empty() ? nullptr : weight.data();
        const TDer1Der2* scratchDersData = scratchDers->data();
        double* resArrData = resArr->data();
        yvector<double> avrg;
        avrg.yresize(1);
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            // TODO(vitekmel): don't forget to remove after weights for pairs added
            double w = weightData == nullptr || error.GetErrorType() == EErrorType::PairwiseError ? 1 : weightData[z];
            UpdateBucket<LeafEstimationType>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<LeafEstimationType>(bucket, iteration, l2Regularizer);
            ExpApproxIf(TError::StoreExpApprox, &avrg);
            resArrData[z] = UpdateApprox<TError::StoreExpApprox>(resArrData[z], avrg[0]);
        }
    }
}

template <typename TError>
void CalcApproxDeltaSimple(const TFold& ff,
                           const yvector<TSplit>& tree,
                           const TError& error,
                           int gradientIterations,
                           ELeafEstimation estimationMethod,
                           float l2Regularizer,
                           TLearnContext* ctx,
                           yvector<yvector<yvector<double>>>* approxDelta,
                           yvector<TIndexType>* ind) {
    auto& indices = *ind;
    approxDelta->resize(ff.BodyTailArr.ysize());

    ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
        const TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];

        yvector<yvector<double>>& resArr = (*approxDelta)[bodyTailId];
        const double initValue = GetNeutralApprox<TError::StoreExpApprox>();
        if (resArr.empty()) {
            resArr.assign(1, yvector<double>(bt.TailFinish, initValue));
        } else {
            Fill(resArr[0].begin(), resArr[0].end(), initValue);
        }

        const int leafCount = GetLeafCount(tree);
        const int scratchSize = Max(!ctx->Params.ApproxOnFullHistory ? 0 : bt.TailFinish - bt.BodyFinish,
                                    error.GetErrorType() == EErrorType::PerObjectError ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT : bt.BodyFinish);
        yvector<TDer1Der2> scratchDers;
        scratchDers.yresize(scratchSize);
        yvector<TSum> buckets(leafCount, TSum(gradientIterations));

        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeafEstimation::Newton) {
                CalcApproxDeltaIterationSimple<ELeafEstimation::Newton>(indices, ff.LearnTarget, ff.LearnWeights, ff.LearnQueryId, ff.LearnQuerySize, bt, error, it, l2Regularizer, ctx,
                                                                        &buckets, &resArr[0], &scratchDers);
            } else {
                CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                CalcApproxDeltaIterationSimple<ELeafEstimation::Gradient>(indices, ff.LearnTarget, ff.LearnWeights, ff.LearnQueryId, ff.LearnQuerySize, bt, error, it, l2Regularizer, ctx,
                                                                          &buckets, &resArr[0], &scratchDers);
            }
        }
    }, 0, ff.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <typename TError>
void CalcApproxDelta(const TFold& ff,
                     const yvector<TSplit>& tree,
                     const TError& error,
                     int gradientIterations,
                     ELeafEstimation estimationMethod,
                     float l2Regularizer,
                     TLearnContext* ctx,
                     yvector<yvector<yvector<double>>>* approxDelta,
                     yvector<TIndexType>* ind) {
    int approxDimension = ff.GetApproxDimension();
    if (approxDimension == 1) {
        CalcApproxDeltaSimple(ff, tree, error, gradientIterations, estimationMethod, l2Regularizer, ctx, approxDelta, ind);
    } else {
        CalcApproxDeltaMulti(ff, tree, error, gradientIterations, estimationMethod, l2Regularizer, ctx, approxDelta, ind);
    }
}

template <ELeafEstimation LeafEstimationType, typename TError>
void CalcLeafValuesIterationSimple(const yvector<TIndexType>& indices,
                                   const yvector<float>& target,
                                   const yvector<float>& weight,
                                   const yvector<ui32>& queriesId,
                                   const yhash<ui32, ui32>& queriesSize,
                                   const yvector<yvector<TCompetitor>>& competitors,
                                   const TError& error,
                                   int iteration,
                                   float l2Regularizer,
                                   TLearnContext* ctx,
                                   yvector<TSum>* buckets,
                                   yvector<double>* approx,
                                   yvector<TDer1Der2>* scratchDers) {
    int leafCount = buckets->ysize();
    int learnSampleCount = approx->ysize();

    if (error.GetErrorType() == EErrorType::PerObjectError) {
        CalcApproxDersRange<LeafEstimationType>(indices.data(), target.data(), weight.data(), approx->data(), /*resArr*/ nullptr,
                                                error, learnSampleCount, iteration, ctx, buckets, scratchDers->data());
    } else if (error.GetErrorType() == EErrorType::PairwiseError) {
        Y_ASSERT(LeafEstimationType == ELeafEstimation::Gradient);
        CalcApproxDersRangePairs(indices, *approx, /*approxDelta*/ yvector<double>(), competitors, error,
                                 learnSampleCount, learnSampleCount, iteration, buckets, scratchDers);
    } else {
        Y_ASSERT(error.GetErrorType() == EErrorType::QuerywiseError);
        Y_ASSERT(LeafEstimationType == ELeafEstimation::Gradient);
        CalcApproxDersQueriesRange(indices, *approx, /*approxDelta*/ yvector<double>(), target, weight, queriesId, queriesSize, error,
                                   learnSampleCount, learnSampleCount, iteration, buckets, scratchDers);
    }

    yvector<double> curLeafValues;
    curLeafValues.yresize(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    UpdateApproxDeltas<TError::StoreExpApprox>(indices, learnSampleCount, ctx, &curLeafValues, approx);
}

template <typename TError>
void CalcLeafValuesSimple(const TTrainData& data,
                          const TFold& ff,
                          const yvector<TSplit>& tree,
                          const TError& error,
                          int gradientIterations,
                          ELeafEstimation estimationMethod,
                          float l2Regularizer,
                          TLearnContext* ctx,
                          yvector<yvector<double>>* leafValues,
                          yvector<TIndexType>* ind) {
    auto& indices = *ind;
    indices = BuildIndices(ff, tree, data, &ctx->LocalExecutor);

    const TFold::TBodyTail& bt = ff.BodyTailArr[0];
    const int leafCount = GetLeafCount(tree);
    const int learnSampleCount = data.LearnSampleCount;

    yvector<yvector<double>> approx(1);
    approx[0].assign(bt.Approx[0].begin(), bt.Approx[0].begin() + learnSampleCount);

    yvector<TSum> buckets(leafCount, gradientIterations);
    const int scratchSize = error.GetErrorType() == EErrorType::PerObjectError
        ? APPROX_BLOCK_SIZE * CB_THREAD_LIMIT
        : learnSampleCount;
    yvector<TDer1Der2> scratchDers(scratchSize);

    for (int it = 0; it < gradientIterations; ++it) {
        if (estimationMethod == ELeafEstimation::Newton) {
            CalcLeafValuesIterationSimple<ELeafEstimation::Newton>(indices, ff.LearnTarget, ff.LearnWeights, ff.LearnQueryId, ff.LearnQuerySize, bt.Competitors,
                                                                   error, it, l2Regularizer, ctx,
                                                                   &buckets, &approx[0], &scratchDers);
        } else {
            CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
            CalcLeafValuesIterationSimple<ELeafEstimation::Gradient>(indices, ff.LearnTarget, ff.LearnWeights, ff.LearnQueryId, ff.LearnQuerySize, bt.Competitors,
                                                                     error, it, l2Regularizer, ctx,
                                                                     &buckets, &approx[0], &scratchDers);
        }
    }

    leafValues->assign(1, yvector<double>(leafCount));
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        for (int it = 0; it < gradientIterations; ++it) {
            (*leafValues)[0][leaf] += (estimationMethod == ELeafEstimation::Newton)
                ? CalcModelNewton(buckets[leaf], it, l2Regularizer)
                : CalcModelGradient(buckets[leaf], it, l2Regularizer);
        }
    }
}

template <typename TError>
void CalcLeafValues(const TTrainData& data,
                    const TFold& ff,
                    const yvector<TSplit>& tree,
                    const TError& error,
                    int gradientIterations,
                    ELeafEstimation estimationMethod,
                    float l2Regularizer,
                    TLearnContext* ctx,
                    yvector<yvector<double>>* leafValues,
                    yvector<TIndexType>* ind) {
    const int approxDimension = ff.GetApproxDimension();
    if (approxDimension == 1) {
        CalcLeafValuesSimple(data, ff, tree, error, gradientIterations, estimationMethod, l2Regularizer, ctx, leafValues, ind);
    } else {
        CalcLeafValuesMulti(data, ff, tree, error, gradientIterations, estimationMethod, l2Regularizer, ctx, leafValues, ind);
    }
}

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
template <typename TError>
void CalcApproxForLeafStruct(const TTrainData& data,
                             const TError& error,
                             int gradientIterations,
                             const yvector<TFold*>& folds,
                             const yvector<TSplit>& tree,
                             ELeafEstimation estimationMethod,
                             float l2Regularizer,
                             TLearnContext* ctx,
                             yvector<yvector<yvector<yvector<double>>>>* approxDelta) { // [foldId][bodyTailId][approxDim][docIdxInPermuted]
    int foldCount = folds.ysize();
    approxDelta->resize(foldCount);
    yvector<yvector<TIndexType>> indices(foldCount);
    ctx->LocalExecutor.ExecRange([&](int foldId) {
        indices[foldId] = BuildIndices(*folds[foldId], tree, data, &ctx->LocalExecutor);
    }, 0, folds.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
    ctx->LocalExecutor.ExecRange([&](int foldId) {
        CalcApproxDelta(*folds[foldId],
                        tree,
                        error,
                        gradientIterations,
                        estimationMethod,
                        l2Regularizer,
                        ctx,
                        &(*approxDelta)[foldId],
                        &indices[foldId]);
    }, 0, folds.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
