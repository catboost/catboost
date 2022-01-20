#include "pairs.h"

#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/exception.h>

#include <util/generic/cast.h>
#include <util/stream/file.h>

using namespace NCB;


void TPairsDataBuilder::Add(i64 groupIdx, i32 winnerIdxInGroup, i32 loserIdxInGroup) {
    Pairs.push_back(
        TPairInGroup{
            SafeIntegerCast<ui32>(groupIdx),
            SafeIntegerCast<ui32>(winnerIdxInGroup),
            SafeIntegerCast<ui32>(loserIdxInGroup)
        }
    );
}

void TPairsDataBuilder::Add(
    i64 groupIdx,
    i32 winnerIdxInGroup,
    i32 loserIdxInGroup,
    float weight
) {
    Pairs.push_back(
        TPairInGroup{
            SafeIntegerCast<ui32>(groupIdx),
            SafeIntegerCast<ui32>(winnerIdxInGroup),
            SafeIntegerCast<ui32>(loserIdxInGroup),
            weight
        }
    );
}

void TPairsDataBuilder::AddToResult(IQuantizedFeaturesDataVisitor* visitor) {
    visitor->SetPairs(TRawPairsData(std::move(Pairs)));
}

void SavePairsInGroupedDsvFormat(
    const NCB::TDataProviderPtr& dataProvider,
    const TString& outputFile
) {

    const TMaybeData<TRawPairsData>& rawPairsData = dataProvider->RawTargetData.GetPairs();
    CB_ENSURE_INTERNAL(rawPairsData, "No pairs data in dataset");
    if (const TGroupedPairsInfo* groupedPairsInfo = std::get_if<TGroupedPairsInfo>(&*rawPairsData)) {
        TOFStream out(outputFile);
        for (const auto& pairInfo : *groupedPairsInfo) {
            out << pairInfo.GroupIdx
                << '\t' << pairInfo.WinnerIdxInGroup
                << '\t' << pairInfo.LoserIdxInGroup
                << '\t' << pairInfo.Weight
                << '\n';
        }
    } else {
        CB_ENSURE_INTERNAL(false, "SavePairsInGroupedDsvFormat expected grouped pairs");
    }
}
