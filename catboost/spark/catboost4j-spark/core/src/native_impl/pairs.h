#pragma once

#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/pairs.h>

#include <util/generic/yexception.h>
#include <util/system/types.h>


namespace NCB {
    class IQuantizedFeaturesDataVisitor;
}


class TPairsDataBuilder {
public:
    void Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup);

    void Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup, float weight);

    void AddToResult(NCB::IQuantizedFeaturesDataVisitor* visitor);
public:
    NCB::TGroupedPairsInfo Pairs;
};

void SavePairsInGroupedDsvFormat(
    const NCB::TDataProviderPtr& dataProvider,
    const TString& outputFile
);
