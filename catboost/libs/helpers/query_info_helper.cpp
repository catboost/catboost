#include "query_info_helper.h"
#include "exception.h"

void UpdateQueriesInfo(const TVector<TGroupId>& queriesId, const TVector<float>& groupWeight, const TVector<ui32>& subgroupId, ui32 beginDoc, ui32 endDoc, TVector<TQueryInfo>* queryInfo) {
    ui32 begin = beginDoc, end = endDoc;
    if (begin == end) {
        return;
    }

    ui32 docIdStart = begin, docIdEnd = end;
    if (queriesId.empty()) {
        queryInfo->emplace_back(docIdStart, docIdEnd);
        if (!subgroupId.empty()) {
            queryInfo->back().SubgroupId = {subgroupId.begin() + docIdStart, subgroupId.begin() + docIdEnd};
        }
        return;
    }

    TGroupId currentQueryId = queriesId[begin];
    ui32 currentQuerySize = 0;
    for (ui32 docId = begin; docId < end; ++docId) {
        if (currentQueryId == queriesId[docId]) {
            ++currentQuerySize;
        } else {
            docIdStart = docId - currentQuerySize;
            docIdEnd = docId;
            queryInfo->emplace_back(docIdStart, docIdEnd);
            if (!subgroupId.empty()) {
                queryInfo->back().SubgroupId = {subgroupId.begin() + docIdStart, subgroupId.begin() + docIdEnd};
            }
            if (!groupWeight.empty()) {
                queryInfo->back().Weight = groupWeight[docIdStart];
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
    if (!groupWeight.empty()) {
        queryInfo->back().Weight = groupWeight[docIdStart];
    }
}

TVector<ui32> GetQueryIndicesForDocs(const TVector<TQueryInfo>& queriesInfo, ui32 learnSampleCount) {
    TVector<ui32> queriesInfoForDocs;
    queriesInfoForDocs.reserve(learnSampleCount);
    for (ui32 queryIndex = 0; queryIndex < queriesInfo.size(); ++queryIndex) {
        queriesInfoForDocs.insert(queriesInfoForDocs.end(), queriesInfo[queryIndex].End - queriesInfo[queryIndex].Begin, queryIndex);
    }
    return queriesInfoForDocs;
}

void UpdateQueriesPairs(const TVector<TPair>& pairs, ui32 beginPair, ui32 endPair, const TVector<size_t>& invertedPermutation, TVector<TQueryInfo>* queryInfo) {
    ui32 begin = beginPair, end = endPair;
    if (begin == end) {
        return;
    }
    TVector<TQueryInfo>& queryInfoRef = *queryInfo;
    TVector<ui32> queriesIndices(queryInfoRef.back().End);
    ui32 currentQueryIndex = 0;
    for (ui32 docId = 0; docId < queriesIndices.size(); ++docId) {
        queriesIndices[docId] = currentQueryIndex;
        if (docId + 1 == queryInfoRef[currentQueryIndex].End) {
            queryInfoRef[currentQueryIndex].Competitors.resize(queryInfoRef[currentQueryIndex].End - queryInfoRef[currentQueryIndex].Begin);
            ++currentQueryIndex;
        }
    }

    for (ui32 pairId = begin; pairId < end; ++pairId) {
        const auto& pair = pairs[pairId];
        ui32 winnerId = invertedPermutation.empty() ? pair.WinnerId : invertedPermutation[pair.WinnerId];
        ui32 loserId = invertedPermutation.empty() ? pair.LoserId : invertedPermutation[pair.LoserId];
        ui32 queryIndex = queriesIndices[winnerId]; // assume that winnerId and loserId belong to the same query
        CB_ENSURE(queryIndex == queriesIndices[loserId], "Both documents in pair should have the same queryId");
        winnerId -= queryInfoRef[queryIndex].Begin;
        loserId -= queryInfoRef[queryIndex].Begin;
        queryInfoRef[queryIndex].Competitors[winnerId].emplace_back(loserId, pair.Weight);
    }
}

TFlatPairsInfo UnpackPairsFromQueries(const TVector<TQueryInfo>& queries) {
    TFlatPairsInfo pairs;
    for (const auto& query : queries) {
        if (query.Competitors.empty()) {
            continue;
        }

        const ui32 begin = query.Begin;
        const ui32 end = query.End;
        for (ui32 winnerId = begin; winnerId < end; ++winnerId) {
            for (const auto& competitor : query.Competitors[winnerId - begin]) {
                pairs.emplace_back(winnerId, competitor.Id + begin, competitor.SampleWeight);
            }
        }
    }
    pairs.shrink_to_fit();
    return pairs;
}
