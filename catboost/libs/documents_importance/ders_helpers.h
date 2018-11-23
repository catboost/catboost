#pragma once

#include <catboost/libs/data/pool.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/vector.h>


void EvaluateDerivatives(
    ELossFunction lossFunction,
    ELeavesEstimation leafEstimationMethod,
    const TVector<double>& approxes,
    const TPool& pool,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
);
