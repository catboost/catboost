#pragma once

#include "learn_context.h"

#include <util/generic/array_ref.h>


namespace NCatboostOptions {
    class TCatBoostOptions;
    class TLossDescription;
}

namespace NPar {
    class ILocalExecutor;
}

void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    NPar::ILocalExecutor* localExecutor
);

void YetiRankRecalculation(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::ILocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
);
