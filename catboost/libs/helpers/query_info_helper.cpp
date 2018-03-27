#include "query_info_helper.h"
#include "exception.h"

void UpdateQueriesInfo(const TVector<TGroupId>& queriesId, const TVector<ui32>& subgroupId, int beginDoc, int endDoc, TVector<TQueryInfo>* queryInfo) {
    int begin = beginDoc, end = endDoc;
    if (begin == end) {
        return;
    }

    int docIdStart = begin, docIdEnd = end;
    if (queriesId.empty()) {
        queryInfo->emplace_back(docIdStart, docIdEnd);
        if (!subgroupId.empty()) {
            queryInfo->back().SubgroupId = {subgroupId.begin() + docIdStart, subgroupId.begin() + docIdEnd};
        }
        return;
    }

    TGroupId currentQueryId = queriesId[begin];
    int currentQuerySize = 0;
    for (int docId = begin; docId < end; ++docId) {
        if (currentQueryId == queriesId[docId]) {
            ++currentQuerySize;
        } else {
            docIdStart = docId - currentQuerySize;
            docIdEnd = docId;
            queryInfo->emplace_back(docIdStart, docIdEnd);
            if (!subgroupId.empty()) {
                queryInfo->back().SubgroupId = {subgroupId.begin() + docIdStart, subgroupId.begin() + docIdEnd};
            }
            currentQuerySize = 1;
            currentQueryId = queriesId[docId];
        }
    }
    docIdStart = end - currentQuerySize;
    docIdEnd = end;
    queryInfo->emplace_back(docIdStart, docIdEnd);
    if (!subgroupId.empty()) {
        queryInfo->back().SubgroupId = {subgroupId.begin() + docIdStart, subgroupId.begin() + docIdEnd};
    }
}

TVector<int> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount) {
    TVector<int> queriesInfoForDocs;
    queriesInfoForDocs.reserve(learnSampleCount);
    for (int queryIndex = 0; queryIndex < queriesInfo.ysize(); ++queryIndex) {
        queriesInfoForDocs.insert(queriesInfoForDocs.end(), queriesInfo[queryIndex].End - queriesInfo[queryIndex].Begin, queryIndex);
    }
    return queriesInfoForDocs;
}

void UpdateQueriesPairs(const TVector<TPair>& pairs, int beginPair, int endPair, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo) {
    int begin = beginPair, end = endPair;
    if (begin == end) {
        return;
    }
    TVector<TQueryInfo>& queryInfoRef = *queryInfo;
    TVector<int> queriesIndices(queryInfoRef.back().End);
    int currentQueryIndex = 0;
    for (int docId = 0; docId < queriesIndices.ysize(); ++docId) {
        queriesIndices[docId] = currentQueryIndex;
        if (docId == queryInfoRef[currentQueryIndex].End - 1) {
            queryInfoRef[currentQueryIndex].Competitors.resize(queryInfoRef[currentQueryIndex].End - queryInfoRef[currentQueryIndex].Begin);
            ++currentQueryIndex;
        }
    }

    for (int pairId = begin; pairId < end; ++pairId) {
        const auto& pair = pairs[pairId];
        int winnerId = invertedPermutation.empty() ? pair.WinnerId : invertedPermutation[pair.WinnerId];
        int loserId = invertedPermutation.empty() ? pair.LoserId : invertedPermutation[pair.LoserId];
        int queryIndex = queriesIndices[winnerId]; // assume that winnerId and loserId belong to the same query
        CB_ENSURE(queryIndex == queriesIndices[loserId], "Both documents in pair should have the same queryId");
        winnerId -= queryInfoRef[queryIndex].Begin;
        loserId -= queryInfoRef[queryIndex].Begin;
        queryInfoRef[queryIndex].Competitors[winnerId].emplace_back(loserId, pair.Weight);
    }
}
