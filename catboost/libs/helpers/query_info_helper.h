#pragma once

#include <catboost/libs/data_types/query.h>

#include <util/generic/vector.h>

void UpdateQueriesInfo(const TVector<ui32>& queriesId, int begin, int end, TVector<TQueryInfo>* queryInfo);

inline void UpdateQueriesInfo(const TVector<ui32>& queriesId, TVector<TQueryInfo>* queryInfo) {
    UpdateQueriesInfo(queriesId, 0, queriesId.ysize(), queryInfo);
}

TVector<int> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount);

void UpdateQueriesPairs(const TVector<TPair>& pairs, int begin, int end, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo);

inline void UpdateQueriesPairs(const TVector<TPair>& pairs, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo) {
    UpdateQueriesPairs(pairs, 0, pairs.ysize(), invertedPermutation, queryInfo);
}
