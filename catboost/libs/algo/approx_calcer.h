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

inline int GetLeafCount(const yvector<TSplit>& tree) {
    return 1 << tree.ysize();
}

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
inline void UpdateApproxBlock(const NPar::TLocalExecutor::TBlockParams& params, int blockIdx, const double* leafValues, const TIndexType* indices, double* resArr) {
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
    ExpApproxIf<StoreExpApprox>(leafValues);

    double* resArrData = resArr->data();
    const TIndexType* indicesData = indices.data();
    const double* leafValuesData = leafValues->data();

    NPar::TLocalExecutor::TBlockParams blockParams(0, docCount);
    blockParams.SetBlockSize(10000).WaitCompletion();

    ctx->LocalExecutor.ExecRange([=] (int blockIdx) {
        UpdateApproxBlock<StoreExpApprox>(blockParams, blockIdx, leafValuesData, indicesData, resArrData);
    }, 0, blockParams.GetBlockCount(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <bool StoreExpApprox>
inline void UpdateApproxDeltasMulti(const yvector<TIndexType>& indices,
                                    int docCount,
                                    yvector<yvector<double>>* leafValues, //leafValues[dimension][bucketId]
                                    yvector<yvector<double>>* resArr) {
    for (int dim = 0; dim < leafValues->ysize(); ++dim) {
        ExpApproxIf<StoreExpApprox>(&(*leafValues)[dim]);
        for (int z = 0; z < docCount; ++z) {
            (*resArr)[dim][z] = UpdateApprox<StoreExpApprox>((*resArr)[dim][z], (*leafValues)[dim][indices[z]]);
        }
    }
}

namespace {
template <ELeafEstimation LeafEstimationType>
inline void UpdateBucket(const TDer1Der2&, double, int, TSum*);

template <>
inline void UpdateBucket<ELeafEstimation::Gradient>(const TDer1Der2& der, double weight, int it, TSum* bucket) {
    bucket->AddDerWeight(der.Der1, weight, it);
}

template <>
inline void UpdateBucket<ELeafEstimation::Newton>(const TDer1Der2& der, double, int it, TSum* bucket) {
    bucket->AddDerDer2(der.Der1, der.Der2, it);
}

template <ELeafEstimation LeafEstimationType>
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
                           TLearnContext* ctx) {
    NPar::TLocalExecutor::TBlockParams blockParams(sampleStart, sampleCount);
    blockParams.SetBlockSize(APPROX_BLOCK_SIZE);
    ctx->LocalExecutor.ExecRange([&](int blockId) {
        const int blockOffset = sampleStart + blockId * blockParams.GetBlockSize(); // espetrov: OK for small datasets
        error.CalcDersRange(blockOffset, Min(blockParams.GetBlockSize(), sampleCount - blockOffset),
                            approx.data(),
                            resArr.data(),
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
                              int sampleCount,
                              const TError& error,
                              int iteration,
                              TLearnContext* ctx,
                              yvector<TSum>* buckets,
                              const double* resArr,
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
                                resArr,
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
void CalcApproxDeltaIteration(const yvector<TIndexType>& indices,
                              const yvector<float>& target,
                              const yvector<float>& weight,
                              const TFold::TBodyTail& bt,
                              const TError& error,
                              int iteration,
                              float l2Regularizer,
                              TLearnContext* ctx,
                              yvector<TSum>* buckets,
                              yvector<double>* resArr,
                              yvector<TDer1Der2>* scratchDers) {
    int leafCount = buckets->ysize();

    CalcApproxDersRange<LeafEstimationType>(indices.data(), target.data(), weight.data(), bt.Approx[0].data(), bt.BodyFinish, error, iteration,
        ctx, buckets, resArr->data(), scratchDers->data());

    // compute mixed model
    yvector<double> curLeafValues(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    // compute tail
    if (ctx->Params.ApproxOnPartialHistory) {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.TailFinish, ctx, &curLeafValues, resArr);
    } else {
        UpdateApproxDeltas<TError::StoreExpApprox>(indices, bt.BodyFinish, ctx, &curLeafValues, resArr);

        CalcShiftedApproxDers(bt.BodyFinish, bt.TailFinish, bt.Approx[0], *resArr, target, weight, error, scratchDers, ctx);
        TSum* bucketsData = buckets->data();
        const TIndexType* indicesData = indices.data();
        const float* weightData = weight.empty() ? nullptr : weight.data();
        const TDer1Der2* scratchDersData = scratchDers->data();
        double* resArrData = resArr->data();
        yvector<double> avrg;
        avrg.yresize(1);
        for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
            TSum& bucket = bucketsData[indicesData[z]];
            double w = weightData == nullptr ? 1 : weightData[z];
            UpdateBucket<LeafEstimationType>(scratchDersData[z - bt.BodyFinish], w, iteration, &bucket);
            avrg[0] = CalcModel<LeafEstimationType>(bucket, iteration, l2Regularizer);
            ExpApproxIf<TError::StoreExpApprox>(&avrg);
            resArrData[z] = UpdateApprox<TError::StoreExpApprox>(resArrData[z], avrg[0]);
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

template <typename TError, typename TCalcModel, typename TAddSampleToBucket>
void CalcApproxDeltaIterationMulti(TCalcModel CalcModel,
                                   TAddSampleToBucket AddSampleToBucket,
                                   const yvector<TIndexType>& indices,
                                   const yvector<float>& target,
                                   const yvector<float>& weight,
                                   const TFold::TBodyTail& bt,
                                   const TError& error,
                                   int iteration,
                                   float l2Regularizer,
                                   yvector<TSumMulti>* buckets,
                                   yvector<yvector<double>>* resArr) {
    int approxDimension = resArr->ysize();
    int leafCount = buckets->ysize();

    yvector<double> curApprox(approxDimension);
    for (int z = 0; z < bt.BodyFinish; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = UpdateApprox<TError::StoreExpApprox>(bt.Approx[dim][z], (*resArr)[dim][z]);
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

    UpdateApproxDeltasMulti<TError::StoreExpApprox>(indices, bt.BodyFinish, &curLeafValues, resArr);

    // compute tail
    for (int z = bt.BodyFinish; z < bt.TailFinish; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = UpdateApprox<TError::StoreExpApprox>(bt.Approx[dim][z], (*resArr)[dim][z]);
        }

        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration, &bucket);

        CalcModel(bucket, iteration, l2Regularizer, &avrg);
        ExpApproxIf<TError::StoreExpApprox>(&avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*resArr)[dim][z] = UpdateApprox<TError::StoreExpApprox>((*resArr)[dim][z], avrg[dim]);
        }
    }
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
    auto& indices = *ind;
    approxDelta->resize(ff.BodyTailArr.ysize());
    const int approxDimension = ff.GetApproxDimension();
    ctx->LocalExecutor.ExecRange([&](int bodyTailId) {
        const TFold::TBodyTail& bt = ff.BodyTailArr[bodyTailId];

        yvector<yvector<double>>& resArr = (*approxDelta)[bodyTailId];
        const double initValue = GetNeutralApprox<TError::StoreExpApprox>();
        if (resArr.empty()) {
            resArr.assign(approxDimension, yvector<double>(bt.TailFinish, initValue));
        } else {
            for (auto& arr : resArr) {
                Fill(arr.begin(), arr.end(), initValue);
            }
        }

        const int leafCount = GetLeafCount(tree);

        if (approxDimension == 1) {
            const int scratchSize = Max(ctx->Params.ApproxOnPartialHistory ? 0 : bt.TailFinish - bt.BodyFinish, APPROX_BLOCK_SIZE * CB_THREAD_LIMIT);
            yvector<TDer1Der2> scratchDers;
            scratchDers.yresize(scratchSize);
            yvector<TSum> buckets(leafCount, TSum(gradientIterations));

            for (int it = 0; it < gradientIterations; ++it) {
                if (estimationMethod == ELeafEstimation::Newton) {
                    CalcApproxDeltaIteration<ELeafEstimation::Newton>(indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer, ctx,
                                             &buckets, &resArr[0], &scratchDers);
                } else {
                    CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                    CalcApproxDeltaIteration<ELeafEstimation::Gradient>(indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer, ctx,
                                             &buckets, &resArr[0], &scratchDers);
                }
            }
        } else {
            yvector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension));
            for (int it = 0; it < gradientIterations; ++it) {
                if (estimationMethod == ELeafEstimation::Newton) {
                    CalcApproxDeltaIterationMulti(CalcModelNewtonMulti, AddSampleToBucketNewtonMulti<TError>,
                                                  indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer,
                                                  &buckets, &resArr);
                }
                else {
                    CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                    CalcApproxDeltaIterationMulti(CalcModelGradientMulti, AddSampleToBucketGradientMulti<TError>,
                                                  indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer,
                                                  &buckets, &resArr);
                }
            }
        }
    }, 0, ff.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
}

template <ELeafEstimation LeafEstimationType, typename TError>
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

    CalcApproxDersRange<LeafEstimationType>(indices.data(), target.data(), weight.data(), approx->data(), learnSampleCount, error, iteration,
      ctx, buckets, /*resArr*/ nullptr, scratchDers->data());

    yvector<double> curLeafValues;
    curLeafValues.yresize(leafCount);
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        curLeafValues[leaf] = CalcModel<LeafEstimationType>((*buckets)[leaf], iteration, l2Regularizer);
    }

    UpdateApproxDeltas<TError::StoreExpApprox>(indices, learnSampleCount, ctx, &curLeafValues, approx);
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

    UpdateApproxDeltasMulti<TError::StoreExpApprox>(indices, learnSampleCount, &curLeafValues, approx);
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
    auto& indices = *ind;
    indices = BuildIndices(ff, tree, data, &ctx->LocalExecutor);

    const TFold::TBodyTail& bt = ff.BodyTailArr[0];
    const int approxDimension = ff.GetApproxDimension();
    const int leafCount = GetLeafCount(tree);

    yvector<yvector<double>> approx(approxDimension);
    for (int dim = 0; dim < approxDimension; ++dim) {
        approx[dim].assign(bt.Approx[dim].begin(), bt.Approx[dim].begin() + data.LearnSampleCount);
    }

    if (approxDimension == 1) {
        yvector<TSum> buckets(leafCount, gradientIterations);
        yvector<TDer1Der2> scratchDers(APPROX_BLOCK_SIZE * CB_THREAD_LIMIT);
        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeafEstimation::Newton) {
                CalcLeafValuesIteration<ELeafEstimation::Newton>(indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer, ctx,
                                        &buckets, &approx[0], &scratchDers);
            } else {
                CB_ENSURE(estimationMethod == ELeafEstimation::Gradient);
                CalcLeafValuesIteration<ELeafEstimation::Gradient>(indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer, ctx,
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
