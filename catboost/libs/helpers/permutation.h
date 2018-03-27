#pragma once

#include "restorable_rng.h"

#include <catboost/libs/data/pool.h>

#include <library/threading/local_executor/local_executor.h>

#include <util/random/shuffle.h>

#include <numeric>

void ApplyPermutation(const TVector<ui64>& permutation, TPool* pool, NPar::TLocalExecutor* localExecutor);
void ApplyPermutationToPairs(const TVector<ui64>& permutation, TVector<TPair>* pairs);
TVector<ui64> CreateOrderByKey(const TVector<ui64>& key);

template<typename IndexType>
TVector<IndexType> InvertPermutation(const TVector<IndexType>& permutation) {
    TVector<IndexType> result(permutation.size());
    for (ui64 i = 0; i < permutation.size(); ++i) {
        result[permutation[i]] = i;
    }
    return result;
}

template<typename TDataType, typename TRandGen>
void Shuffle(const TVector<TGroupId>& queryId, TRandGen& rand, TVector<TDataType>* indices) {
    if (queryId.empty()) {
        Shuffle(indices->begin(), indices->end(), rand);
        return;
    }

    TVector<std::pair<int, int>> queryStartAndSize;
    int docsToPermute = indices->ysize();
    for (int docIdx = 0; docIdx < docsToPermute; ++docIdx) {
        if (docIdx == 0 || queryId[docIdx] != queryId[docIdx - 1]) {
            queryStartAndSize.emplace_back(docIdx, 1);
        }
        else {
            queryStartAndSize.back().second++;
        }
    }
    Shuffle(queryStartAndSize.begin(), queryStartAndSize.end(), rand);

    int idxInResult = 0;
    for (int queryIdx = 0; queryIdx < queryStartAndSize.ysize(); queryIdx++) {
        const auto& query = queryStartAndSize[queryIdx];
        int initialStart = query.first;
        int resultStart = idxInResult;
        int size = query.second;
        for (int doc = 0; doc < size; doc++) {
            (*indices)[resultStart + doc] = initialStart + doc;
        }
        Shuffle(indices->begin() + resultStart, indices->begin() + resultStart + size, rand);
        idxInResult += size;
    }
}
