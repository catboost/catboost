#pragma once

#include <catboost/libs/helpers/matrix.h>
#include <catboost/libs/helpers/restorable_rng.h>

#include <catboost/private/libs/algo/learn_context.h>
#include <catboost/private/libs/algo_helpers/error_functions.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <library/cpp/threading/local_executor/local_executor.h>


void CalcExactLeafDeltasMulti(
    const NCatboostOptions::TLossDescription& lossDescription,
    const TVector<TIndexType>& indices, // not used if leaf count == 1
    const size_t sampleCount,
    const TVector<TVector<double>>& approxes,
    TConstArrayRef<TConstArrayRef<float>> targets,
    TConstArrayRef<float> weight,
    size_t leafCount,
    NPar::ILocalExecutor* localExecutor,
    TVector<TVector<double>>* leafDeltas);

void CalcLeafValuesMulti(
    int leafCount,
    const IDerCalcer& error,
    const TVector<TQueryInfo>& queryInfo, // not used if leaf count == 1
    const TVector<TIndexType>& indices, // not used if leaf count == 1
    const TVector<TConstArrayRef<float>>& label,
    TConstArrayRef<float> weight,
    double sumWeight,
    int l2RegSampleCount, // used for regularization only
    int sampleCount,
    TLearnContext* ctx,
    TVector<TVector<double>>* sumLeafDeltas, // [dim][leafIdx]
    TVector<TVector<double>>* approx);
