#pragma once

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/query.h>

#include <util/generic/vector.h>

void UpdateQueriesInfo(const TVector<TGroupId>& queriesId, const TVector<float>& groupWeight, const TVector<ui32>& subgroupId, ui32 beginDoc, ui32 endDoc, TVector<TQueryInfo>* queryInfo);

TVector<ui32> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, ui32 learnSampleCount);

void UpdateQueriesPairs(const TVector<TPair>& pairs, ui32 beginPair, ui32 endPair, const TVector<ui32>& invertedPermutation, TVector<TQueryInfo>* queryInfo);

inline void UpdateQueriesPairs(const TVector<TPair>& pairs, const TVector<ui32>& invertedPermutation, TVector<TQueryInfo>* queryInfo) {
    UpdateQueriesPairs(pairs, 0, pairs.ysize(), invertedPermutation, queryInfo);
}

TFlatPairsInfo UnpackPairsFromQueries(const TVector<TQueryInfo>& queries);
