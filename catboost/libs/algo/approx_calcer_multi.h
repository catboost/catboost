#pragma once

#include "approx_util.h"
#include "index_calcer.h"
#include "online_predictor.h"

template <bool StoreExpApprox>
inline void UpdateApproxDeltasMulti(const yvector<TIndexType>& indices,
                                    int docCount,
                                    yvector<yvector<double>>* leafValues, //leafValues[dimension][bucketId]
                                    yvector<yvector<double>>* resArr) {
    for (int dim = 0; dim < leafValues->ysize(); ++dim) {
        ExpApproxIf(StoreExpApprox, &(*leafValues)[dim]);
        for (int z = 0; z < docCount; ++z) {
            (*resArr)[dim][z] = UpdateApprox<StoreExpApprox>((*resArr)[dim][z], (*leafValues)[dim][indices[z]]);
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
        ExpApproxIf(TError::StoreExpApprox, &avrg);
        for (int dim = 0; dim < approxDimension; ++dim) {
            (*resArr)[dim][z] = UpdateApprox<TError::StoreExpApprox>((*resArr)[dim][z], avrg[dim]);
        }
    }
}


template <typename TError>
void CalcApproxDeltaMulti(const TFold& ff,
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
        yvector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension));
        for (int it = 0; it < gradientIterations; ++it) {
            if (estimationMethod == ELeafEstimation::Newton) {
                CalcApproxDeltaIterationMulti(CalcModelNewtonMulti, AddSampleToBucketNewtonMulti<TError>,
                                              indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer,
                                              &buckets, &resArr);
            } else {
                Y_ASSERT(estimationMethod == ELeafEstimation::Gradient);
                CalcApproxDeltaIterationMulti(CalcModelGradientMulti, AddSampleToBucketGradientMulti<TError>,
                                              indices, ff.LearnTarget, ff.LearnWeights, bt, error, it, l2Regularizer,
                                              &buckets, &resArr);
            }
        }
    }, 0, ff.BodyTailArr.ysize(), NPar::TLocalExecutor::WAIT_COMPLETE);
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
void CalcLeafValuesMulti(const TTrainData& data,
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

    yvector<TSumMulti> buckets(leafCount, TSumMulti(approxDimension));
    for (int it = 0; it < gradientIterations; ++it) {
        if (estimationMethod == ELeafEstimation::Newton) {
            CalcLeafValuesIterationMulti(CalcModelNewtonMulti, AddSampleToBucketNewtonMulti<TError>,
                                         indices, ff.LearnTarget, ff.LearnWeights, error, it, l2Regularizer,
                                         &buckets, &approx);
        } else {
            Y_ASSERT(estimationMethod == ELeafEstimation::Gradient);
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
