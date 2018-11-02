#pragma once

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/query.h>

#include <util/generic/fwd.h>

void UpdateQueriesInfo(
    TConstArrayRef<TGroupId> queriesId,
    TConstArrayRef<float> groupWeight,
    TConstArrayRef<ui32> subgroupId,
    ui32 beginDoc,
    ui32 endDoc,
    TVector<TQueryInfo>* queryInfo);

TVector<ui32> GetQueryIndicesForDocs(TConstArrayRef<TQueryInfo> queriesInfo, ui32 learnSampleCount);

void UpdateQueriesPairs(
    TConstArrayRef<TPair> pairs,
    ui32 beginPair,
    ui32 endPair,
    TConstArrayRef<ui32> invertedPermutation,
    TVector<TQueryInfo>* queryInfo);

void UpdateQueriesPairs(
    TConstArrayRef<TPair> pairs,
    TConstArrayRef<ui32> invertedPermutation,
    TVector<TQueryInfo>* queryInfo);

TFlatPairsInfo UnpackPairsFromQueries(TConstArrayRef<TQueryInfo> queries);
