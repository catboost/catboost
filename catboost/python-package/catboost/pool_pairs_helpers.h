#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/pairs.h>
#include <catboost/libs/helpers/exception.h>

namespace NCB {
    // Return stored pairs in the current Pool's row coordinates. Subsetting
    // already remaps raw target pairs; grouped storage only needs expansion
    // from group-relative indices using this provider's current group bounds.
    inline TVector<TPair> GetPythonPoolPairs(const TDataProvider& data) {
        const auto& rawPairs = data.RawTargetData.GetPairs();
        if (!rawPairs) {
            return {};
        }
        if (const auto* flat = std::get_if<TFlatPairsInfo>(&*rawPairs)) {
            return *flat;
        }
        const auto& grouped = std::get<TGroupedPairsInfo>(*rawPairs);
        TVector<TPair> result;
        result.reserve(grouped.size());
        for (const auto& pair : grouped) {
            CB_ENSURE(pair.GroupIdx < data.ObjectsGrouping->GetGroupCount(),
                "Stored Pool pair refers to an invalid group");
            const auto bounds = data.ObjectsGrouping->GetGroup(pair.GroupIdx);
            CB_ENSURE(pair.WinnerIdxInGroup < bounds.GetSize() && pair.LoserIdxInGroup < bounds.GetSize(),
                "Stored Pool pair refers to an invalid object within its group");
            result.emplace_back(bounds.Begin + pair.WinnerIdxInGroup,
                                bounds.Begin + pair.LoserIdxInGroup,
                                pair.Weight);
        }
        return result;
    }
}
