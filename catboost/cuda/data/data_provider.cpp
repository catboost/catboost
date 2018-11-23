#include "data_provider.h"
#include <util/stream/file.h>

namespace NCatboostCuda {
    void TDataProvider::DumpBordersToFileInMatrixnetFormat(const TString& file) {
        TOFStream out(file);
        for (auto& feature : Features) {
            if (feature->GetType() == EFeatureValuesType::BinarizedFloat) {
                auto binarizedFeaturePtr = dynamic_cast<const TBinarizedFloatValuesHolder*>(feature.Get());
                auto nanMode = binarizedFeaturePtr->GetNanMode();

                for (const auto& border : binarizedFeaturePtr->GetBorders()) {
                    out << binarizedFeaturePtr->GetId() << "\t" << ToString<double>(border);
                    if (nanMode != ENanMode::Forbidden) {
                        out << "\t" << nanMode;
                    }
                    out << Endl;
                }
            }
        }
    }

    void TDataProvider::FillQueryPairs(const TVector<TPair>& pairs) {
        CB_ENSURE(QueryIds.size(), "Error: provide query ids");
        THashMap<TGroupId, ui32> queryOffsets;
        for (ui32 doc = 0; doc < QueryIds.size(); ++doc) {
            const auto queryId = QueryIds[doc];
            if (!queryOffsets.has(queryId)) {
                queryOffsets[queryId] = doc;
            }
        }
        for (const auto& pair : pairs) {
            CB_ENSURE(QueryIds[pair.LoserId] == QueryIds[pair.WinnerId], "Error: pair documents should be in one query");
            const auto queryId = QueryIds[pair.LoserId];
            TPair localPair = pair;
            ui32 offset = queryOffsets[queryId];
            localPair.WinnerId -= offset;
            localPair.LoserId -= offset;
            QueryPairs[queryId].push_back(localPair);
        }
    }

    void TDataProvider::GeneratePairs()  {
        CB_ENSURE(QueryIds.size(), "Error: provide query ids");
        THashMap<TGroupId, ui32> queryOffsets;
        THashMap<TGroupId, ui32> querySizes;
        for (ui32 doc = 0; doc < QueryIds.size(); ++doc) {
            const auto queryId = QueryIds[doc];
            if (!queryOffsets.has(queryId)) {
                queryOffsets[queryId] = doc;
            }
            querySizes[queryId]++;
        }
        ui32 pairCount = 0;
        for (const auto& query : queryOffsets) {
            const auto qid = query.first;
            const ui32 offset = queryOffsets[qid];
            const ui32 size = querySizes[qid];

            for (ui32 i = 0; i < size; ++i) {
                for (ui32 j = 0; j < i; ++j) {
                    if (Targets[offset + i] != Targets[offset + j]) {
                        TPair pair;
                        const bool isFirstBetter = Targets[offset + i] > Targets[offset + j];
                        pair.WinnerId = isFirstBetter ? i : j;
                        pair.LoserId = isFirstBetter ? j : i;
                        pair.Weight = std::abs(Targets[offset + i] - Targets[offset + j]);
                        QueryPairs[qid].push_back(pair);
                        ++pairCount;
                    }
                }
            }
        }
        CATBOOST_DEBUG_LOG << pairCount << " pairs generated" << Endl;
    }
}
