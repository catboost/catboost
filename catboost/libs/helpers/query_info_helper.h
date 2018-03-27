#pragma once

#include <catboost/libs/data_types/groupid.h>
#include <catboost/libs/data_types/query.h>

#include <util/generic/vector.h>

void UpdateQueriesInfo(const TVector<TGroupId>& queriesId, const TVector<ui32>& subgroupId, int beginDoc, int endDoc, TVector<TQueryInfo>* queryInfo);

TVector<int> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount);

void UpdateQueriesPairs(const TVector<TPair>& pairs, int beginPair, int endPair, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo);

inline void UpdateQueriesPairs(const TVector<TPair>& pairs, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo) {
    UpdateQueriesPairs(pairs, 0, pairs.ysize(), invertedPermutation, queryInfo);
}
