#include "query_info_helper.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

TVector<ui32> GetQueryIndicesForDocs(
    const TConstArrayRef<TQueryInfo> queriesInfo,
    const ui32 learnSampleCount)
{
    TVector<ui32> queriesInfoForDocs;
    queriesInfoForDocs.reserve(learnSampleCount);
    for (size_t queryIndex = 0; queryIndex < queriesInfo.size(); ++queryIndex) {
        queriesInfoForDocs.insert(
            queriesInfoForDocs.end(),
            queriesInfo[queryIndex].End - queriesInfo[queryIndex].Begin,
            queryIndex);
    }
    return queriesInfoForDocs;
}

TFlatPairsInfo UnpackPairsFromQueries(TConstArrayRef<TQueryInfo> queries) {
    size_t pairsCount = 0;
    for (const auto& query : queries) {
        if (query.Competitors.empty()) {
            continue;
        }

        const ui32 begin = query.Begin;
        const ui32 end = query.End;
        for (ui32 winnerId = begin; winnerId < end; ++winnerId) {
            pairsCount += query.Competitors[winnerId - begin].size();
        }
    }

    TFlatPairsInfo pairs;
    pairs.reserve(pairsCount);

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
