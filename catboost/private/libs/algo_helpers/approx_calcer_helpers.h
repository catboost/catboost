#pragma once

#include "ders_holder.h"
#include "online_predictor.h"

#include <catboost/libs/metrics/metric.h>
#include <catboost/private/libs/options/enums.h>
#include <catboost/private/libs/options/oblivious_tree_options.h>

#include <util/generic/ptr.h>

static constexpr int APPROX_BLOCK_SIZE = 500;

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

void CreateBacktrackingObjective(
    NCatboostOptions::TLossDescription metricDescriptions,
    const TMaybe<TCustomMetricDescriptor>& customMetric,
    const NCatboostOptions::TObliviousTreeLearnerOptions& treeOptions,
    int approxDimension,
    bool* haveBacktrackingObjective,
    double* minimizationSign,
    TVector<THolder<IMetric>>* lossFunction
);

double GetMinimizeSign(const THolder<IMetric>& metric);
