#pragma once

#include "error_functions.h"
#include "online_predictor.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/restrictions.h>

void UpdateApproxDeltasMulti(
    TConstArrayRef<TIndexType> indices, // not used if leaf count == 1
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas, // [dimension][leafId]
    TVector<TVector<double>>* approxDeltas,
    NPar::ILocalExecutor* localExecutor
);

void SetApproxDeltasMulti(
    TConstArrayRef<TIndexType> indices, // not used if leaf count == 1
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas, // [dimension][leafId]
    TVector<TVector<double>>* approxDeltas,
    NPar::ILocalExecutor* localExecutor
);

inline void AddDersRangeMulti(
    TConstArrayRef<TIndexType> leafIndices,
    TConstArrayRef<float> target,
    TConstArrayRef<float> weight,
    TConstArrayRef<TVector<double>> approx, // [dimensionIdx][columnIdx]
    TConstArrayRef<TVector<double>> approxDeltas, // [dimensionIdx][columnIdx]
    const IDerCalcer& error,
    int rowBegin,
    int rowEnd,
    bool isUpdateWeight,
    TArrayRef<TSumMulti> leafDers // [dimensionIdx]
);

void CalcLeafDersMulti(
    const TVector<TIndexType>& indices,
    TConstArrayRef<TConstArrayRef<float>> target,
    TConstArrayRef<float> weight,
    const TVector<TVector<double>>& approx,
    const TVector<TVector<double>>& approxDeltas,
    const IDerCalcer& error,
    int sampleCount,
    bool isUpdateWeight,
    ELeavesEstimation estimationMethod,
    NPar::ILocalExecutor* localExecutor,
    TVector<TSumMulti>* leafDers
);

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDers,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<TVector<double>>* curLeafValues // [approxDim][leafIdx]
);
