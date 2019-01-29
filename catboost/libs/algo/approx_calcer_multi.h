#pragma once

#include "index_calcer.h"
#include "online_predictor.h"
#include "approx_updater_helpers.h"
#include "error_functions.h"

void UpdateApproxDeltasMulti(
    bool storeExpApprox,
    const TVector<TIndexType>& indices,
    int docCount,
    TVector<TVector<double>>* leafValues, //leafValues[dimension][bucketId]
    TVector<TVector<double>>* resArr
);

inline void AddSampleToBucketNewtonMulti(
    const IDerCalcer& error,
    const TVector<double>& approx,
    float target,
    double weight,
    int,
    TVector<double>* curDer,
    THessianInfo* curDer2,
    TSumMulti* bucket
) {
    Y_ASSERT(curDer != nullptr && curDer2 != nullptr);
    error.CalcDersMulti(approx, target, weight, curDer, curDer2);
    bucket->AddDerDer2(*curDer, *curDer2);
}

inline void AddSampleToBucketGradientMulti(
    const IDerCalcer& error,
    const TVector<double>& approx,
    float target,
    double weight,
    int iteration,
    TVector<double>* curDer,
    THessianInfo* /*curDer2*/,
    TSumMulti* bucket
) {
    Y_ASSERT(curDer != nullptr);
    error.CalcDersMulti(approx, target, weight, curDer, nullptr);
    bucket->AddDerWeight(*curDer, weight, iteration);
}

template <typename TAddSampleToBucket>
void UpdateBucketsMulti(
    TAddSampleToBucket AddSampleToBucket,
    const TVector<TIndexType>& indices,
    const TVector<float>& target,
    const TVector<float>& weight,
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& resArr,
    const IDerCalcer& error,
    int sampleCount,
    int iteration,
    TVector<TSumMulti>* buckets
) {
    const int approxDimension = resArr.ysize();
    Y_ASSERT(approxDimension > 0);
    TVector<double> curApprox(approxDimension);
    TVector<double> bufferDer(approxDimension);
    THessianInfo bufferDer2(approxDimension, error.GetHessianType());
    for (int z = 0; z < sampleCount; ++z) {
        for (int dim = 0; dim < approxDimension; ++dim) {
            curApprox[dim] = approx.empty() ? resArr[dim][z] : UpdateApprox(error.GetIsExpApprox(), approx[dim][z], resArr[dim][z]);
        }
        TSumMulti& bucket = (*buckets)[indices[z]];
        AddSampleToBucket(error, curApprox, target[z], weight.empty() ? 1 : weight[z], iteration,
                          &bufferDer, &bufferDer2, &bucket);
    }
}

template <typename TCalcMethodDelta>
void CalcMixedModelMulti(
    TCalcMethodDelta CalcMethodDelta,
    const TVector<TSumMulti>& buckets,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<TVector<double>>* curLeafValues
) {
    const int leafCount = buckets.ysize();
    TVector<double> avrg;
    for (int leaf = 0; leaf < leafCount; ++leaf) {
        CalcMethodDelta(buckets[leaf], l2Regularizer, sumAllWeights, docCount, &avrg);
        for (int dim = 0; dim < avrg.ysize(); ++dim) {
            (*curLeafValues)[dim][leaf] = avrg[dim];
        }
    }
}

void CalcApproxDeltaMulti(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    int leafCount,
    const IDerCalcer& error,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* approxDelta,
    TVector<TVector<double>>* sumLeafValues
);

void CalcLeafValuesMulti(
    int leafCount,
    const IDerCalcer& error,
    const TFold& ff,
    const TVector<TIndexType>& indices,
    TLearnContext* ctx,
    TVector<TVector<double>>* leafValues
);
