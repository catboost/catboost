#pragma once

#include "online_predictor.h"
#include "ders_holder.h"

#include <catboost/libs/options/enums.h>

template <ELeavesEstimation LeafEstimationType>
inline void UpdateBucket(const TDers&, double, int, TSum*);

template <>
inline void UpdateBucket<ELeavesEstimation::Gradient>(const TDers& der, double weight, int it, TSum* bucket) {
    bucket->AddDerWeight(der.Der1, weight, it);
}

template <>
inline void UpdateBucket<ELeavesEstimation::Newton>(const TDers& der, double, int it, TSum* bucket) {
    bucket->AddDerDer2(der.Der1, der.Der2, it);
}

template <ELeavesEstimation LeafEstimationType>
inline double CalcModel(const TSum&, int gradientIteration, float l2Regularizer, double sumAllWeights, int allDocCount);

template <>
inline double CalcModel<ELeavesEstimation::Gradient>(const TSum& ss, int gradientIteration, float l2Regularizer, double sumAllWeights, int allDocCount) {
    return CalcModelGradient(ss, gradientIteration, l2Regularizer, sumAllWeights, allDocCount);
}

template <>
inline double CalcModel<ELeavesEstimation::Newton>(const TSum& ss, int gradientIteration, float l2Regularizer, double sumAllWeights, int allDocCount) {
    return CalcModelNewton(ss, gradientIteration, l2Regularizer, sumAllWeights, allDocCount);
}
