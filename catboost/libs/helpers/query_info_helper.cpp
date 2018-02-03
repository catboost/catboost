#include "query_info_helper.h"

void UpdateQueriesInfo(const TVector<ui32>& queriesId, int begin, int end, TVector<TQueryInfo>* queryInfo) {
    if (begin == end) {
        return;
    }
    ui32 currentQueryId = queriesId[begin];
    int currentQuerySize = 0;
    for (int docId = begin; docId < end; ++docId) {
        if (currentQueryId == queriesId[docId]) {
            ++currentQuerySize;
        } else {
            queryInfo->push_back({docId - currentQuerySize, docId});
            currentQuerySize = 1;
            currentQueryId = queriesId[docId];
        }
    }
    queryInfo->push_back({end - currentQuerySize, end});
}

TVector<TQueryEndInfo> GetQueryEndInfo(const TVector<TQueryInfo>& queriesInfo, int learnSampleCount) {
    TVector<TQueryEndInfo> queriesInfoForDocs;
    queriesInfoForDocs.reserve(learnSampleCount);
    for (int queryIndex = 0; queryIndex < queriesInfo.ysize(); ++queryIndex) {
        queriesInfoForDocs.insert(
            queriesInfoForDocs.end(),
            queriesInfo[queryIndex].End- queriesInfo[queryIndex].Begin,
            {queriesInfo[queryIndex].End, queryIndex}
        );
    }
    return queriesInfoForDocs;
}

