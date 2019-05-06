#pragma once

#include "learn_context.h"

#include <util/generic/array_ref.h>


namespace NCatboostOptions {
    class TCatBoostOptions;
    class TLossDescription;
}

namespace NPar {
    class TLocalExecutor;
}

void UpdatePairsForYetiRank(
    TConstArrayRef<double> approxes,
    TConstArrayRef<float> relevances,
    const NCatboostOptions::TLossDescription& lossDescription,
    ui64 randomSeed,
    int queryBegin,
    int queryEnd,
    TVector<TQueryInfo>* queriesInfo,
    NPar::TLocalExecutor* localExecutor
);

void YetiRankRecalculation(
    const TFold& ff,
    const TFold::TBodyTail& bt,
    const NCatboostOptions::TCatBoostOptions& params,
    ui64 randomSeed,
    NPar::TLocalExecutor* localExecutor,
    TVector<TQueryInfo>* recalculatedQueriesInfo,
    TVector<float>* recalculatedPairwiseWeights
);
