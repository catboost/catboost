#pragma once

#include <catboost/libs/data_types/query.h>

#include <util/generic/vector.h>

void UpdateQueriesInfo(const TVector<ui32>& queriesId, const TVector<ui32>& subgroupId, int begin, int end, TVector<TQueryInfo>* queryInfo);

TVector<int> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount);

void UpdateQueriesPairs(const TVector<TPair>& pairs, int begin, int end, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo);
