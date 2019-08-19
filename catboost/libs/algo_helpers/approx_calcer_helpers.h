#pragma once

#include "ders_holder.h"
#include "online_predictor.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/libs/options/enums.h>

#include <util/generic/ptr.h>

template <ELeavesEstimation LeafEstimationType>
inline void AddMethodDer(const TDers&, double, bool, TSum*);

template <>
inline void AddMethodDer<ELeavesEstimation::Gradient>(const TDers& der, double weight, bool updateWeight, TSum* bucket) {
    bucket->AddDerWeight(der.Der1, weight, updateWeight);
}

template <>
inline void AddMethodDer<ELeavesEstimation::Newton>(const TDers& der, double, bool, TSum* bucket) {
    bucket->AddDerDer2(der.Der1, der.Der2);
}

template <ELeavesEstimation LeafEstimationType>
inline double CalcMethodDelta(const TSum& ss, float l2Regularizer, double sumAllWeights, int allDocCount);

template <>
inline double CalcMethodDelta<ELeavesEstimation::Gradient>(
    const TSum& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount)
{
    return CalcDeltaGradient(ss, l2Regularizer, sumAllWeights, allDocCount);
}

template <>
inline double CalcMethodDelta<ELeavesEstimation::Newton>(
    const TSum& ss,
    float l2Regularizer,
    double sumAllWeights,
    int allDocCount)
{
    return CalcDeltaNewton(ss, l2Regularizer, sumAllWeights, allDocCount);
}
