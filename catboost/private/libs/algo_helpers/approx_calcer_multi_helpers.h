#pragma once

#include "error_functions.h"
#include "online_predictor.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/restrictions.h>

void UpdateApproxDeltasMulti(
    const TVector<TIndexType>& indices,
    int docCount,
    TConstArrayRef<TVector<double>> leafDeltas, //leafDeltas[dimension][leafId]
    TVector<TVector<double>>* approxDeltas,
    NPar::TLocalExecutor* localExecutor
);

void UpdateApproxDeltasMulti(
    const TVector<TIndexType>& indices,
    int docCount,
    TConstArrayRef<double> leafDeltas, //leafDeltas[dimension]
    TVector<TVector<double>>* approxDeltas,
    NPar::TLocalExecutor* localExecutor
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
    NPar::TLocalExecutor* localExecutor,
    TVector<TSumMulti>* leafDers
);

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDers,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<TVector<double>>* curLeafValues
);

void CalcLeafDeltasMulti(
    const TVector<TSumMulti>& leafDer,
    ELeavesEstimation estimationMethod,
    float l2Regularizer,
    double sumAllWeights,
    int docCount,
    TVector<double>* curLeafValues
);
