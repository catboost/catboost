#pragma once

#include <catboost/private/libs/algo_helpers/ders_holder.h>
#include <catboost/private/libs/algo_helpers/leaf_statistics.h>
#include <catboost/private/libs/options/catboost_options.h>

#include <util/generic/array_ref.h>

class IDerCalcer;
struct TRestorableFastRng64;
class TLearnContext;

void CalcLeafValues(
    const IDerCalcer& error,
    TLeafStatistics* statistics,
    TLearnContext* ctx,
    TArrayRef<TDers> weightedDers = {});

void AssignLeafValues(
    const TVector<TLeafStatistics>& statistics, // [dim][leafId]
    TVector<TVector<double>>* treeValues);
