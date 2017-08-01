#pragma once

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

inline int GetLeafCount(const TTensorStructure3& tree) {
    return 1 << tree.SelectedSplits.ysize();
}

template <int vectorWidth>
inline void UpdateApproxKernel(const double* leafValues, const TIndexType* indices, double* resArr) {
    Y_ASSERT(vectorWidth == 4);
    const TIndexType idx0 = indices[0];
    const TIndexType idx1 = indices[1];
    const TIndexType idx2 = indices[2];
    const TIndexType idx3 = indices[3];
    const double value0 = leafValues[idx0];
    const double value1 = leafValues[idx1];
    const double value2 = leafValues[idx2];
    const double value3 = leafValues[idx3];
    resArr[0] += value0;
    resArr[1] += value1;
    resArr[2] += value2;
    resArr[3] += value3;
}

inline void UpdateApproxBlock(const NPar::TLocalExecutor::TBlockParams& params, int blockIdx, const double* leafValues, const TIndexType* indices, double* resArr) {
    const int blockStart = blockIdx * params.GetBlockSize();
    const int nextBlockStart = Min<ui64>(blockStart + params.GetBlockSize(), params.LastId);
    constexpr int vectorWidth = 4;
    int doc;
    for (doc = blockStart; doc + vectorWidth <= nextBlockStart; doc += vectorWidth) {
        UpdateApproxKernel<vectorWidth>(leafValues, indices + doc, resArr + doc);
    }
    for (; doc < nextBlockStart; ++doc) {
        resArr[doc] += leafValues[indices[doc]];
    }
}

inline void UpdateApproxDeltas(const yvector<double>& leafValues,
                               const yvector<TIndexType>& indices,
                               int docCount,
                               TLearnContext* ctx,
                               yvector<double>* resArr) {
    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues.data();

    NPar::TLocalExecutor::TBlockParams blockParams(0, docCount);
    blockParams.SetBlockSize(10000).WaitCompletion();

    ctx->LocalExecutor.ExecRange([=] (int blockIdx) {
        UpdateApproxBlock(blockParams, blockIdx, leafValuesData, indicesData, resArrData);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

inline void UpdateApproxDeltasMulti(const yvector<yvector<double>>& leafValues, //leafValues[dimension][bucketId]
                                    const yvector<TIndexType>& indices,
                                    int docCount,
                                    yvector<yvector<double>>* resArr) {
    for (int dim = 0; dim < leafValues.ysize(); ++dim) {
        for (int z = 0; z < docCount; ++z) {
            (*resArr)[dim][z] += leafValues[dim][indices[z]];
        }
    }
}

namespace {
template <ELeafEstimation type>
inline void UpdateBucket(const TDer1Der2&, double, int, TSum*);

template <>
inline void UpdateBucket<ELeafEstimation::Gradient>(const TDer1Der2& der, double weight, int it, TSum* bucket) {
    bucket->AddDerWeight(der.Der1, weight, it);
}

template <>
inline void UpdateBucket<ELeafEstimation::Newton>(const TDer1Der2& der, double, int it, TSum* bucket) {
    bucket->AddDerDer2(der.Der1, der.Der2, it);
}

template <ELeafEstimation type>
inline double CalcModel(const TSum&, int, float);

template <>
inline double CalcModel<ELeafEstimation::Gradient>(const TSum& ss, int gradientIteration, float l2Regularizer) {
    return CalcModelGradient(ss, gradientIteration, l2Regularizer);
}

template <>
inline double CalcModel<ELeafEstimation::Newton>(const TSum& ss, int gradientIteration, float l2Regularizer) {
    return CalcModelNewton(ss, gradientIteration, l2Regularizer);
}

constexpr int APPROX_BLOCK_SIZE = 500;

template <typename TError>
void CalcShiftedApproxDers(int sampleStart,
                           int sampleCount,
                           const yvector<double>& approx,
                           const yvector<double>& resArr,
                           const yvector<float>& target,
                           const yvector<float>& weight,
                           const TError& error,
                           yvector<TDer1Der2>* scratchDers,
                           yvector<double>* scratchApprox,
                           TLearnContext* ctx) {
    NPar::TLocalExecutor::TBlockParams blockParams(sampleStart, sampleCount);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor.ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        const double* approxData = approx.data();
        const double* resArrData = resArr.data();
        double* shiftedApproxData = scratchApprox->data();
        NPar::TLocalExecutor::BlockedLoopBody(blockParams, [shiftedApproxData, approxData, resArrData, sampleStart] (int z) {
            shiftedApproxData[z - sampleStart] = approxData[z] + resArrData[z];
        })(blockId);
        error.CalcDersRange(Min(blockParams.GetBlockSize(), sampleCount - blockOffset), shiftedApproxData + blockOffset - sampleStart,
                            target.data() + blockOffset, weight.empty() ? nullptr : weight.data() + blockOffset,
                            scratchDers->data() + blockOffset - sampleStart);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}
} // anonymous namespace

template <ELeafEstimation type, typename TError>
void CalcApproxDeltaRange(const TIndexType* indices,
                              const float* target,
                              const float* weight,
                              const double* approx,
                              int sampleCount,
                              const TError& error,
                              int iteration,
                              TLearnContext* ctx,
                              yvector<TSum>* buckets,
                              const double* resArr,
                              TDer1Der2* scratchDers,
                              double* scratchApprox) {
    const int leafCount = buckets->ysize();

    NPar::TLocalExecutor::TBlockParams blockParams(0, sampleCount);
    blockParams.SetBlockCount(CB_THREAD_LIMIT);

    yvector<yvector<TDer1Der2>> blockBucketDers(blockParams.GetBlockCount(), yvector<TDer1Der2>(leafCount, TDer1Der2{/*Der1*/0.0, /*Der2*/0.0}));
    yvector<TDer1Der2>* blockBucketDersData = blockBucketDers.data();
    yvector<yvector<double>> blockBucketSumWeights(blockParams.GetBlockCount(), yvector<double>(leafCount, 0));
    yvector<double>* blockBucketSumWeightsData = blockBucketSumWeights.data();

    ctx->LocalExecutor.ExecRange([=](int blockId) {
        constexpr int innerBlockSize = APPROX_BLOCK_SIZE;
        double* shiftApprox = scratchApprox + innerBlockSize * blockId;
        TDer1Der2* approxDer = scratchDers + innerBlockSize * blockId;

        const int blockStart = blockId * blockParams.GetBlockSize();
        const int nextBlockStart = Min(sampleCount, blockStart + blockParams.GetBlockSize());

        TDer1Der2* bucketDers = blockBucketDersData[blockId].data();
        double* bucketSumWeights = blockBucketSumWeightsData[blockId].data();

        for (int innerBlockStart = blockStart; innerBlockStart < nextBlockStart; innerBlockStart += innerBlockSize) {
            const int nextInnerBlockStart = Min(nextBlockStart, innerBlockStart + innerBlockSize);
            if (resArr != nullptr) {
                for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                    shiftApprox[z - innerBlockStart] = approx[z] + resArr[z]; // each thread touches at most blockSize in shiftedApprox
                }
            }
            error.CalcDersRange(nextInnerBlockStart - innerBlockStart, resArr != nullptr ? shiftApprox : approx + innerBlockStart,
                                target + innerBlockStart, weight == nullptr ? nullptr : weight + innerBlockStart,
                                approxDer);
            for (int z = innerBlockStart; z < nextInnerBlockStart; ++z) {
                TDer1Der2& ders = bucketDers[indices[z]];
                ders.Der1 += approxDer[z - innerBlockStart].Der1;
                ders.Der2 += approxDer[z - innerBlockStart].Der2;
                bucketSumWeights[indices[z]] += weight != nullptr ? weight[z] : 1;
            }
        }
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);

    for (int leafId = 0; leafId < leafCount; ++leafId) {
        for (int blockId = 0; blockId < blockParams.GetBlockCount(); ++blockId) {
            if (blockBucketSumWeights[blockId][leafId] > FLT_EPSILON) {
                UpdateBucket<type>(blockBucketDers[blockId][leafId], blockBucketSumWeights[blockId][leafId], iteration, &(*buckets)[leafId]);
            }
        }
    }
}

template <ELeafEstimation type, typename TError>
void CalcApproxDeltaIteration(const yvector<TIndexType>& indices,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              const TFold::TMixTail& mt,
                              const TError& error,
                              int iteration,
                              float l2Regularizer,
                              TLearnContext* ctx,
                              yvector<TSum>* buckets,
                              yvector<double>* resArr,
                              yvector<TDer1Der2>* scratchDers,
                              yvector<double>* scratchApprox) {
    int leafCount = buckets->ysize();

    CalcApproxDeltaRange<type, TError>(indices.data(), target.data(), weight.data(), mt.Approx[0].data(), mt.MixCount, error, iteration,
        ctx, buckets, resArr->data(), scratchDers->data(), scratchApprox->data());

    // compute mixed model
    yvector<double> curLeafValues(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<type>((*buckets)[leaf], iteration, l2Regularizer);
    }

    // compute tail
    if (ctx->Params.ApproxOnAllHistory) {
        UpdateApproxDeltas(curLeafValues, indices, mt.TailFinish, ctx, resArr);
    } else {
        UpdateApproxDeltas(curLeafValues, indices, mt.MixCount, ctx, resArr);

        CalcShiftedApproxDers(mt.MixCount, mt.TailFinish, mt.Approx[0], *resArr, target, weight, error, scratchDers, scratchApprox, ctx);
        TSum* bucketsData = buckets->data();
        const TIndexType* indicesData = indices.data();
        const float* weightData = weight.empty() ? nullptr : weight.data();
        const TDer1Der2* scratchDersData = scratchDers->data();
        double* resArrData = resArr->data();
        for (int z = mt.MixCount; z < mt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weightData == nullptr ? 1 : weightData[z];
            UpdateBucket<type>(scratchDersData[z - mt.MixCount], w, iteration, &bucket);
            double avrg = CalcModel<type>(bucket, iteration, l2Regularizer);
            resArrData[z] += avrg;
        }
    }
}

template <typename TError>
void AddSampleToBucketNewtonMulti(const TError& error, const yvector<double>& approx,
                                  float target, double weight, int iteration, TSumMulti* bucket) {
    const int approxDimension = approx.ysize();
    yvector<double> curDer(approxDimension);
    TArray2D<double> curDer2(approxDimension, approxDimension);
    error.CalcDersMulti(approx, target, weight, &curDer, &curDer2);
    bucket->AddDerDer2(curDer, curDer2, iteration);
}

template <typename TError>
void AddSampleToBucketGradientMulti(const TError& error, const yvector<double>& approx,
                                    float target, double weight, int iteration, TSumMulti* bucket) {
    yvector<double> curDer(approx.ysize());
    error.CalcDersMulti(approx, target, weight, &curDer, nullptr);
    bucket->AddDerWeight(curDer, weight, iteration);
}

template <typename TCalcModel, typename TAddSampleToBucket, typename TError>
void CalcApproxDeltaIterationMulti(TCalcModel CalcModel,
                                   TAddSampleToBucket AddSampleToBucket,
                                   const yvector<TIndexType>& indices,
                                   const yvector<float>& target,
                                   const yvector<float>& weight,
                                   const TFold::TMixTail& mt,
                                   const TError& error,
                                   int iteration,
                                   float l2Regularizer,
                                   yvector<TSumMulti>* buckets,
                                   yvector<yvector<double>>* resArr) {
    int approxDimension = resArr->ysize();
    int leafCount = buckets->ysize();

    yvector<double> curApprox(approxDimension);
    for (int z = 0; z < mt.MixCount; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = mt.Approx[dim][z] + (*resArr)[dim][z];
        }
        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration, &bucket);
    }

    // compute mixed model
    yvector<yvector<double>> curLeafValues(approxDimension, yvector<double>(leafCount));
    yvector<double> avrg(approxDimension);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        CalcModel((*buckets)[leaf], iteration, l2Regularizer, &avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            curLeafValues[dim][leaf] = avrg[dim];
        }
    }

    UpdateApproxDeltasMulti(curLeafValues, indices, mt.MixCount, resArr);

    // compute tail
    for (int z = mt.MixCount; z < mt.TailFinish; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = mt.Approx[dim][z] + (*resArr)[dim][z];
        }

        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration, &bucket);

        CalcModel(bucket, iteration, l2Regularizer, &avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*resArr)[dim][z] += avrg[dim];
        }
    }
}

template <typename TError>
void CalcApproxDelta(const TFold& ff,
                     const TTensorStructure3& ts,
                     const TError& error,
                     int gradientIterations,
                     ELeafEstimation estimationMethod,
                     float l2Regularizer,
                     TLearnContext* ctx,
                     yvector<yvector<yvector<double>>>* approxDelta,
                     yvector<TIndexType>* ind) {
    auto& indices = *ind;
    approxDelta->resize(ff.MixTailArr.ysize());
    const int approxDimension = ff.GetApproxDimension();
    ctx->LocalExecutor.ExecRange([&](int mixTailId) {
        const TFold::TMixTail& mt = ff.MixTailArr[mixTailId];

        yvector<yvector<double>>& resArr = (*approxDelta)[mixTailId];
        if (resArr.empty()) {
            resArr.assign(approxDimension, yvector<double>(mt.TailFinish));
        } else {
            for (auto& arr : resArr) {
                Clear(&arr, mt.TailFinish);
            }
        }

        const int leafCount = GetLeafCount(ts);

        if (approxDimension == 1) {
            const int scratchSize = Max(ctx->Params.ApproxOnAllHistory ? 0 : mt.TailFinish - mt.MixCount, APPROX_BLOCK_SIZE * CB_THREAD_LIMIT);
            yvector<TDer1Der2> scratchDers;
            scratchDers.yresize(scratchSize);
            yvector<double> scratchApprox;
            scratchApprox.yresize(scratchSize);
            yvector<TSum> buckets(leafCount, TSum(gradientIterations));

            for (int it = 0; it < gradientIterations; ++it) {
                if (estimationMethod == ELeafEstimation::Newton) {
                    CalcApproxDeltaIteration<ELeafEstimation::Newton, TError>(indices, ff.LearnTarget, ff.LearnWeights, mt, error, it, l2Regularizer, ctx,
                                             &buckets, &resArr[0], &scratchDers, &scratchApprox);
                } else {
                    CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                    CalcApproxDeltaIteration<ELeafEstimation::Gradient, TError>(indices, ff.LearnTarget, ff.LearnWeights, mt, error, it, l2Regularizer, ctx,
                                             &buckets, &resArr[0], &scratchDers, &scratchApprox);
                }
            }
        } else {
            yvector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension));
            for (int it = 0; it < gradientIterations; ++it) {
                if (estimationMethod == ELeafEstimation::Newton) {
                    CalcApproxDeltaIterationMulti(CalcModelNewtonMulti, AddSampleToBucketNewtonMulti<TError>,
                                                  indices, ff.LearnTarget, ff.LearnWeights, mt, error, it, l2Regularizer,
                                                  &buckets, &resArr);
                }
                else {
                    CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                    CalcApproxDeltaIterationMulti(CalcModelGradientMulti, AddSampleToBucketGradientMulti<TError>,
                                                  indices, ff.LearnTarget, ff.LearnWeights, mt, error, it, l2Regularizer,
                                                  &buckets, &resArr);
                }
            }
        }
    }, 0, ff.MixTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <ELeafEstimation type, typename TError>
void CalcLeafValuesIteration(const yvector<TIndexType>& indices,
                             const yvector<float>& target,
                             const yvector<float>& weight,
                             const TError& error,
                             int iteration,
                             float l2Regularizer,
                             TLearnContext* ctx,
                             yvector<TSum>* buckets,
                             yvector<double>* approx,
                             yvector<TDer1Der2>* scratchDers) {
    int leafCount = buckets->ysize();
    int learnSampleCount = approx->ysize();

    CalcApproxDeltaRange<type, TError>(indices.data(), target.data(), weight.data(), approx->data(), learnSampleCount, error, iteration,
      ctx, buckets, /*resArr*/nullptr, scratchDers->data(), /*scratchApprox*/nullptr);

    yvector<double> curLeafValues(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<type>((*buckets)[leaf], iteration, l2Regularizer);
    }

    UpdateApproxDeltas(curLeafValues, indices, learnSampleCount, ctx, approx);
}


template <typename TCalcModel, typename TAddSampleToBucket, typename TError>
void CalcLeafValuesIterationMulti(TCalcModel CalcModel,
                                  TAddSampleToBucket AddSampleToBucket,
                                  const yvector<TIndexType>& indices,
                                  const yvector<float>& target,
                                  const yvector<float>& weight,
                                  const TError& error,
                                  int iteration,
                                  float l2Regularizer,
                                  yvector<TSumMulti>* buckets,
                                  yvector<yvector<double>>* approx) {
    int leafCount = buckets->ysize();
    int approxDimension = approx->ysize();
    int learnSampleCount = (*approx)[0].ysize();

    yvector<double> curApprox(approxDimension);
    for (int z = 0; z < learnSampleCount; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = (*approx)[dim][z];
        }

        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration, &bucket);
    }

    yvector<yvector<double>> curLeafValues(approxDimension, yvector<double>(leafCount));
    yvector<double> avrg(approxDimension);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        CalcModel((*buckets)[leaf], iteration, l2Regularizer, &avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            curLeafValues[dim][leaf] = avrg[dim];
        }
    }

    UpdateApproxDeltasMulti(curLeafValues, indices, learnSampleCount, approx);
}

template <typename TError>
void CalcLeafValues(const TTrainData& data,
                    const TFold& ff,
                    const TTensorStructure3& ts,
                    const TError& error,
                    int gradientIterations,
                    ELeafEstimation estimationMethod,
                    float l2Regularizer,
                    TLearnContext* ctx,
                    yvector<yvector<double>>* leafValues,
                    yvector<TIndexType>* ind) {
    auto& indices = *ind;
    indices = BuildIndices(ff, ts, data, &ctx->LocalExecutor);

    const TFold::TMixTail& mt = ff.MixTailArr[0];
    const int approxDimension = ff.GetApproxDimension();
    const int leafCount = GetLeafCount(ts);

    yvector<yvector<double>> approx(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        approx[dim].assign(mt.Approx[dim].begin(), mt.Approx[dim].begin() + data.LearnSampleCount);
    }

    if (approxDimension == 1) {
        yvector<TSum> buckets(leafCount, gradientIterations);
        yvector<TDer1Der2> scratchDers(APPROX_BLOCK_SIZE * CB_THREAD_LIMIT);
        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeafEstimation::Newton) {
                CalcLeafValuesIteration<ELeafEstimation::Newton, TError>(indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer, ctx,
                                        &buckets, &approx[0], &scratchDers);
            } else {
                CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                CalcLeafValuesIteration<ELeafEstimation::Gradient, TError>(indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer, ctx,
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
    } else {
        yvector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension));
        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeafEstimation::Newton) {
                CalcLeafValuesIterationMulti(CalcModelNewtonMulti, AddSampleToBucketNewtonMulti<TError>,
                                             indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer,
                                             &buckets, &approx);
            }
            else {
                CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                CalcLeafValuesIterationMulti(CalcModelGradientMulti, AddSampleToBucketGradientMulti<TError>,
                                             indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer,
                                             &buckets, &approx);
            }
        }

        yvector<double> avrg(approxDimension);
        leafValues->assign(approxDimension, yvector<double>(leafCount));
        for (int leaf = 0; leaf < leafCount; ++leaf) {
            for (int it = 0; it < gradientIterations; ++it) {
                if (estimationMethod == ELeafEstimation::Newton) {
                    CalcModelNewtonMulti(buckets[leaf], it, l2Regularizer, &avrg);
                } else {
                    CalcModelGradientMulti(buckets[leaf], it, l2Regularizer, &avrg);
                }
                for (int dim = 0; dim < approxDimension; ++dim) {
                    (*leafValues)[dim][leaf] += avrg[dim];
                }
            }
        }
    }
}

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
template <typename TError>
void CalcApproxForLeafStruct(const TTrainData& data,
                             const TError& error,
                             int gradientIterations,
                             const yvector<TFold*>& folds,
                             const TTensorStructure3& ts,
                             ELeafEstimation estimationMethod,
                             float l2Regularizer,
                             TLearnContext* ctx,
                             yvector<yvector<yvector<yvector<double>>>>* approxDelta) { // [foldId][mixTailId][approxDim][docIdxInPermuted]
    int foldCount = folds.ysize();
    approxDelta->resize(foldCount);
    yvector<yvector<TIndexType>> indices(foldCount);
    ctx->LocalExecutor.ExecRange([&](int foldId) {
        indices[foldId] = BuildIndices(*folds[foldId], ts, data, &ctx->LocalExecutor);
    }, 0, folds.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
    ctx->LocalExecutor.ExecRange([&](int foldId) {
        CalcApproxDelta(*folds[foldId],
                        ts,
                        error,
                        gradientIterations,
                        estimationMethod,
                        l2Regularizer,
                        ctx,
                        &(*approxDelta)[foldId],
                        &indices[foldId]);
    }, 0, folds.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

// output is permuted (learnSampleCount samples are permuted by LearnPermutation, test is indexed directly)
template <typename TError>
void CalcApprox(const TTrainData& data,
                const TError& error,
                int gradientIterations,
                TFold* fold,
                const TTensorStructure3& ts,
                ELeafEstimation estimationMethod,
                float l2Regularizer,
                TLearnContext* ctx,
                yvector<yvector<double>>* approxDelta, // [approxDim][docIdxInPermuted]
                yvector<yvector<double>>* leafValues) {
    yvector<TIndexType> indices;
    CalcLeafValues(data,
                   *fold,
                   ts,
                   error,
                   gradientIterations,
                   estimationMethod,
                   l2Regularizer,
                   ctx,
                   leafValues,
                   &indices);
    const int sampleCount = data.GetSampleCount();
    // TODO(annaveronika): remove approxDelta
    const int approxDimension = fold->GetApproxDimension();
    if (approxDelta->empty()) {
        approxDelta->assign(approxDimension, yvector<double>(sampleCount));
    }
    for (int dim = 0; dim < approxDimension; ++dim) {
        for (int i = 0; i < sampleCount; ++i) {
            (*approxDelta)[dim][i] = (*leafValues)[dim][indices[i]];
        }
    }
}
