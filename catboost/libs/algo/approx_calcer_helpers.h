#pragma once

#include "online_predictor.h"
#include "ders_holder.h"

#include <catboost/libs/options/enums.h>

template <ELeavesEstimation LeafEstimationType>
inline void AddMethodDer(const TDers&, double, int, TSum*);

template <>
inline void AddMethodDer<ELeavesEstimation::Gradient>(const TDers& der, double weight, int it, TSum* bucket) {
    bucket->AddDerWeight(der.Der1, weight, it);
}

template <>
inline void AddMethodDer<ELeavesEstimation::Newton>(const TDers& der, double, int, TSum* bucket) {
    bucket->AddDerDer2(der.Der1, der.Der2);
}

template <ELeavesEstimation LeafEstimationType>
inline double CalcMethodDelta(const TSum& ss, float l2Regularizer, double sumAllWeights, int allDocCount);

template <>
inline double CalcMethodDelta<ELeavesEstimation::Gradient>(const TSum& ss, float l2Regularizer, double sumAllWeights, int allDocCount) {
    return CalcDeltaGradient(ss, l2Regularizer, sumAllWeights, allDocCount);
}

template <>
inline double CalcMethodDelta<ELeavesEstimation::Newton>(const TSum& ss, float l2Regularizer, double sumAllWeights, int allDocCount) {
    return CalcDeltaNewton(ss, l2Regularizer, sumAllWeights, allDocCount);
}
