#pragma once

#include <catboost/private/libs/options/enums.h>

#include <util/generic/fwd.h>
#include <util/generic/vector.h>


void EvaluateDerivatives(
    ELossFunction lossFunction,
    ELeavesEstimation leafEstimationMethod,
    const TVector<double>& approxes,
    TConstArrayRef<float> target,
    TVector<double>* firstDerivatives,
    TVector<double>* secondDerivatives,
    TVector<double>* thirdDerivatives
);
