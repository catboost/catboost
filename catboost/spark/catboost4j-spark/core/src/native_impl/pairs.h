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
    void Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup) throw(yexception);

    void Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup, float weight) throw(yexception);

    void AddToResult(NCB::IQuantizedFeaturesDataVisitor* visitor) throw(yexception);
public:
    NCB::TGroupedPairsInfo Pairs;
};

void SavePairsInGroupedDsvFormat(
    const NCB::TDataProviderPtr& dataProvider,
    const TString& outputFile
) throw(yexception);
