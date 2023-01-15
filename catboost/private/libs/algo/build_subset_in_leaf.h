#pragma once

#include "learn_context.h"

#include <catboost/private/libs/algo_helpers/leaf_statistics.h>
#include <catboost/private/libs/options/restrictions.h>

#include <library/cpp/threading/local_executor/local_executor.h>

TVector<TLeafStatistics> BuildSubset(
    TConstArrayRef<TIndexType> leafIndices,
    int leafCount,
    TLearnContext* ctx);
