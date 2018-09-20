#include "objects_grouping.h"

#include <util/generic/xrange.h>


using namespace NCB;


void NCB::CheckIsConsecutive(TConstArrayRef<TGroupBounds> groups) {
    ui32 expectedBegin = 0;
    for (auto i : xrange(groups.size())) {
        CB_ENSURE(
            groups[i].Begin == expectedBegin,
            "groups[" << i << "].Begin is not equal to expected (" << expectedBegin << ')'
        );
        expectedBegin = groups[i].End;
    }
}


TObjectsGroupingSubset NCB::GetSubset(
    TObjectsGroupingPtr objectsGrouping,
    TArraySubsetIndexing<ui32>&& groupsSubset
) {
    using TSubsetVariantType = typename TArraySubsetIndexing<ui32>::TBase;

    if (objectsGrouping->IsTrivial()) {
        return TObjectsGroupingSubset(
            (groupsSubset.Index() == TSubsetVariantType::template TagOf<TFullSubset<ui32>>()) ?
                objectsGrouping : MakeIntrusive<TObjectsGrouping>(groupsSubset.Size()),
            std::move(groupsSubset)
        );
    } else {
        THolder<TArraySubsetIndexing<ui32>> objectsSubset;
        TVector<TGroupBounds> subsetGroupBounds;

        switch (groupsSubset.Index()) {
            case TSubsetVariantType::template TagOf<TFullSubset<ui32>>():
                objectsSubset = MakeHolder<TArraySubsetIndexing<ui32>>(
                    TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                );
                return TObjectsGroupingSubset(
                    objectsGrouping,
                    std::move(groupsSubset),
                    MakeHolder<TArraySubsetIndexing<ui32>>(
                        TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                    )
                );
            case TSubsetVariantType::template TagOf<TRangesSubset<ui32>>(): {
                    const auto& groupsSubsetBlocks =
                        groupsSubset.template Get<TRangesSubset<ui32>>().Blocks;

                    ui32 srcObjectCount = objectsGrouping->GetObjectCount();
                    auto nontrivialSrcGroups = objectsGrouping->GetNonTrivialGroups();

                    subsetGroupBounds.reserve(groupsSubset.Size());

                    TVector<TSubsetBlock<ui32>> objectsSubsetBlocks;
                    objectsSubsetBlocks.reserve(groupsSubsetBlocks.size());

                    ui32 objectsDstBegin = 0;
                    for (const auto& groupSubsetBlock : groupsSubsetBlocks) {
                        objectsSubsetBlocks.emplace_back(
                            TIndexRange<ui32>(
                                nontrivialSrcGroups[groupSubsetBlock.SrcBegin].Begin,
                                (groupSubsetBlock.SrcEnd == nontrivialSrcGroups.size() ?
                                    srcObjectCount : nontrivialSrcGroups[groupSubsetBlock.SrcEnd].Begin)
                            ),
                            objectsDstBegin
                        );

                        for (auto srcGroupIdx : xrange(groupSubsetBlock.SrcBegin, groupSubsetBlock.SrcEnd)) {
                            subsetGroupBounds.emplace_back(
                                objectsDstBegin,
                                objectsDstBegin + nontrivialSrcGroups[srcGroupIdx].GetSize()
                            );
                            objectsDstBegin += subsetGroupBounds.back().GetSize();
                        }
                    }

                    objectsSubset = MakeHolder<TArraySubsetIndexing<ui32>>(
                        TRangesSubset<ui32>(objectsDstBegin, std::move(objectsSubsetBlocks))
                    );
                }
                break;
            case TSubsetVariantType::template TagOf<TIndexedSubset<ui32>>(): {
                    const auto& groupsSubsetIndices =
                        groupsSubset.template Get<TIndexedSubset<ui32>>();

                    auto nontrivialSrcGroups = objectsGrouping->GetNonTrivialGroups();

                    subsetGroupBounds.reserve(groupsSubset.Size());

                    TVector<TSubsetBlock<ui32>> objectsSubsetBlocks;
                    objectsSubsetBlocks.reserve(groupsSubsetIndices.size());

                    ui32 objectsDstBegin = 0;
                    for (auto srcGroupIdx : groupsSubsetIndices) {
                        objectsSubsetBlocks.emplace_back(nontrivialSrcGroups[srcGroupIdx], objectsDstBegin);

                        subsetGroupBounds.emplace_back(
                            objectsDstBegin,
                            objectsDstBegin + nontrivialSrcGroups[srcGroupIdx].GetSize()
                        );

                        objectsDstBegin += subsetGroupBounds.back().GetSize();
                    }

                    objectsSubset = MakeHolder<TArraySubsetIndexing<ui32>>(
                        TRangesSubset<ui32>(objectsDstBegin, std::move(objectsSubsetBlocks))
                    );
                }
                break;
        }

        return TObjectsGroupingSubset(
            MakeIntrusive<TObjectsGrouping>(std::move(subsetGroupBounds), true),
            std::move(groupsSubset),
            std::move(objectsSubset)
        );
    }
}
