#pragma once

#include "fold.h"

#include <catboost/private/libs/algo_helpers/online_predictor.h>
#include <catboost/private/libs/options/restrictions.h>

class IDerCalcer;
class TLearnContext;


namespace NPar {
    class TLocalExecutor;
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
