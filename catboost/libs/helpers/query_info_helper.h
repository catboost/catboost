#pragma once

#include <catboost/private/libs/data_types/groupid.h>
#include <catboost/private/libs/data_types/query.h>

#include <util/generic/fwd.h>

TVector<ui32> GetQueryIndicesForDocs(const TConstArrayRef<TQueryInfo> queriesInfo, const ui32 learnSampleCount);
TFlatPairsInfo UnpackPairsFromQueries(TConstArrayRef<TQueryInfo> queries);
