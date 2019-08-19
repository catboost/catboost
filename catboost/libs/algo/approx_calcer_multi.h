#pragma once

#include "fold.h"

#include <catboost/libs/algo_helpers/online_predictor.h>
#include <catboost/libs/options/restrictions.h>

class IDerCalcer;
class TLearnContext;


namespace NPar {
    class TLocalExecutor;
}

void UpdateApproxDeltasMulti(
    const TVector<TIndexType>& indices,
    int docCount,
    NPar::TLocalExecutor* localExecutor,
    TVector<TVector<double>>* leafDeltas, //leafDeltas[dimension][leafId]
    TVector<TVector<double>>* approxDeltas
);

void CalcLeafDersMulti(
    const TVector<TIndexType>& indices,
    const TVector<float>& target,
    const TVector<float>& weight,
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
    TVector<TVector<double>>* leafDeltas
);
