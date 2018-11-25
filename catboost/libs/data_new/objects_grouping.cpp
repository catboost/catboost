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
    TArraySubsetIndexing<ui32>&& groupsSubset,
    EObjectsOrder groupSubsetOrder
) {
    using TSubsetVariantType = typename TArraySubsetIndexing<ui32>::TBase;

    if (objectsGrouping->IsTrivial()) {
        return TObjectsGroupingSubset(
            (groupsSubset.Index() == TSubsetVariantType::template TagOf<TFullSubset<ui32>>()) ?
                objectsGrouping : MakeIntrusive<TObjectsGrouping>(groupsSubset.Size()),
            std::move(groupsSubset),
            groupSubsetOrder
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
                    groupSubsetOrder,
                    MakeHolder<TArraySubsetIndexing<ui32>>(
                        TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                    ),
                    groupSubsetOrder
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
            groupSubsetOrder,
            std::move(objectsSubset),
            groupSubsetOrder
        );
    }
}


TObjectsGroupingSubset NCB::GetGroupingSubsetFromObjectsSubset(
    TObjectsGroupingPtr objectsGrouping,
    TArraySubsetIndexing<ui32>&& objectsSubset,
    EObjectsOrder subsetOrder
) {
    if (objectsGrouping->IsTrivial()) {
        return GetSubset(objectsGrouping, std::move(objectsSubset), subsetOrder);
    } else {
        auto nontrivialSrcGroups = objectsGrouping->GetNonTrivialGroups();

        TIndexedSubset<ui32> groupsSubset;
        ui32 subsetCurrentGroupOffset = 0;

        constexpr TStringBuf INVARIANT_MESSAGE = AsStringBuf(
            " subset groups invariant (if group is present in the subset all it's member objects"
            " must present and be in the same order within a group). This constraint might be"
            " relaxed in the future.");

        objectsSubset.ForEach(
            [&](ui32 idx, ui32 srcIdx) {
                if (groupsSubset.empty()) {
                    groupsSubset.push_back(objectsGrouping->GetGroupIdxForObject(srcIdx));
                    subsetCurrentGroupOffset = 1;
                } else {
                    const auto& lastGroup = nontrivialSrcGroups[groupsSubset.back()];
                    if (subsetCurrentGroupOffset == lastGroup.GetSize()) {
                        if (subsetOrder == EObjectsOrder::Ordered) {
                            CB_ENSURE(
                                srcIdx >= lastGroup.End,
                                "subset's object #" << idx << " (source index #" << srcIdx << ") violates"
                                " ordered subset invariant"
                            );
                        }

                        // new group must be started
                        groupsSubset.push_back(objectsGrouping->GetGroupIdxForObject(srcIdx));
                        subsetCurrentGroupOffset = 1;
                    } else {
                        CB_ENSURE(
                            srcIdx == (lastGroup.Begin + subsetCurrentGroupOffset),
                            "subset's object #" << idx << " (source index #" << srcIdx << ") violates"
                            << INVARIANT_MESSAGE
                        );
                        ++subsetCurrentGroupOffset;
                    }
                }
            }
        );
        if (!groupsSubset.empty()) {
            ui32 srcGroupSize = nontrivialSrcGroups[groupsSubset.back()].GetSize();
            CB_ENSURE(
                srcGroupSize == subsetCurrentGroupOffset,
                "Subset's last group size (" << subsetCurrentGroupOffset << ") is less than "
                "corresponding source group size (" << srcGroupSize << "). It violates" << INVARIANT_MESSAGE
            );
        }
        return GetSubset(objectsGrouping, TArraySubsetIndexing<ui32>(std::move(groupsSubset)), subsetOrder);
    }
}
