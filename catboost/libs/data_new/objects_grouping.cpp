#include "objects_grouping.h"

#include <catboost/libs/helpers/permutation.h>

#include <util/generic/xrange.h>
#include <util/random/shuffle.h>

#include <numeric>


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
            ::HoldsAlternative<TFullSubset<ui32>>(groupsSubset) ?
                objectsGrouping : MakeIntrusive<TObjectsGrouping>(groupsSubset.Size()),
            std::move(groupsSubset),
            groupSubsetOrder
        );
    } else {
        TMaybe<TArraySubsetIndexing<ui32>> objectsSubset;
        TVector<TGroupBounds> subsetGroupBounds;

        switch (groupsSubset.index()) {
            case TVariantIndexV<TFullSubset<ui32>, TSubsetVariantType>:
                objectsSubset = MakeMaybe<TArraySubsetIndexing<ui32>>(
                    TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                );
                return TObjectsGroupingSubset(
                    objectsGrouping,
                    std::move(groupsSubset),
                    groupSubsetOrder,
                    MakeMaybe<TArraySubsetIndexing<ui32>>(
                        TFullSubset<ui32>(objectsGrouping->GetObjectCount())
                    ),
                    groupSubsetOrder
                );
            case TVariantIndexV<TRangesSubset<ui32>, TSubsetVariantType>: {
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

                    objectsSubset = MakeMaybe<TArraySubsetIndexing<ui32>>(
                        TRangesSubset<ui32>(objectsDstBegin, std::move(objectsSubsetBlocks))
                    );
                }
                break;
            case TVariantIndexV<TIndexedSubset<ui32>, TSubsetVariantType>: {
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

                    objectsSubset = MakeMaybe<TArraySubsetIndexing<ui32>>(
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


TObjectsGroupingSubset NCB::Shuffle(
    TObjectsGroupingPtr objectsGrouping,
    ui32 permuteBlockSize,
    TRestorableFastRng64* rand
) {
    const ui32 objectCount = objectsGrouping->GetObjectCount();

    TIndexedSubset<ui32> indices;
    indices.yresize(objectCount);

    if (objectsGrouping->IsTrivial()) {
        if (permuteBlockSize == 1) {
            CreateShuffledIndices(objectCount, rand, &indices);
        } else {
            const ui32 blocksCount = (objectCount + permuteBlockSize - 1) / permuteBlockSize;
            TVector<ui32> blockedPermute;
            CreateShuffledIndices(blocksCount, rand, &blockedPermute);

            ui32 currentIdx = 0;
            for (ui32 i = 0; i < blocksCount; ++i) {
                const ui32 blockStartIdx = blockedPermute[i] * permuteBlockSize;
                const ui32 blockEndIndx = Min(blockStartIdx + permuteBlockSize, objectCount);
                for (ui32 j = blockStartIdx; j < blockEndIndx; ++j) {
                    indices[currentIdx + j - blockStartIdx] = j;
                }
                currentIdx += blockEndIndx - blockStartIdx;
            }
        }
        return TObjectsGroupingSubset(
            objectsGrouping,
            TArraySubsetIndexing<ui32>(std::move(indices)),
            EObjectsOrder::RandomShuffled
        );
    } else {
        CB_ENSURE_INTERNAL(permuteBlockSize == 1, "permuteBlockSize must be 1 if groups are present");

        TIndexedSubset<ui32> groupPermute;
        CreateShuffledIndices(objectsGrouping->GetGroupCount(), rand, &groupPermute);

        const TConstArrayRef<TGroupBounds> srcGroupsBounds = objectsGrouping->GetNonTrivialGroups();

        TVector<TGroupBounds> dstGroupBounds;
        dstGroupBounds.yresize(objectsGrouping->GetGroupCount());

        ui32 idxInResult = 0;
        for (ui32 queryIdx = 0; queryIdx < (ui32)groupPermute.size(); queryIdx++) {
            TGroupBounds srcGroupBounds = srcGroupsBounds[groupPermute[queryIdx]];
            ui32 initialStart = srcGroupBounds.Begin;
            ui32 resultStart = idxInResult;
            ui32 size = srcGroupBounds.GetSize();
            dstGroupBounds[queryIdx] = TGroupBounds(idxInResult, idxInResult + size);
            for (ui32 doc = 0; doc < size; doc++) {
                indices[resultStart + doc] = initialStart + doc;
            }
            Shuffle(indices.begin() + resultStart, indices.begin() + resultStart + size, *rand);
            idxInResult += size;
        }

        return TObjectsGroupingSubset(
            MakeIntrusive<TObjectsGrouping>(std::move(dstGroupBounds), true),
            TArraySubsetIndexing<ui32>(std::move(groupPermute)),
            EObjectsOrder::RandomShuffled,
            MakeMaybe<TArraySubsetIndexing<ui32>>(std::move(indices)),
            EObjectsOrder::RandomShuffled
        );
    }
}


TVector<TArraySubsetIndexing<ui32>> NCB::Split(
    const TObjectsGrouping& objectsGrouping,
    ui32 partCount,
    bool oldCvStyle
) {
    const ui32 objectCount = objectsGrouping.GetObjectCount();

    TVector<TArraySubsetIndexing<ui32>> result;

    if (objectsGrouping.IsTrivial()) {
        ui32 currentEnd = 0;
        for (ui32 part = 0; part < partCount; ++part) {
            TSubsetBlock<ui32> block;
            if (oldCvStyle) {
                block.SrcBegin = currentEnd;
                block.SrcEnd = objectCount * (part + 1) / partCount;
                currentEnd = block.SrcEnd;
            } else {
                InitElementRange(
                    part,
                    partCount,
                    objectCount,
                    &block.SrcBegin,
                    &block.SrcEnd
                );
            }
            const ui32 blockSize = block.GetSize();
            CB_ENSURE(blockSize > 0, "Not enough objects for splitting into requested amount of parts");
            block.DstBegin = 0;
            result.push_back(
                TArraySubsetIndexing<ui32>(
                    TRangesSubset<ui32>(blockSize, TVector<TSubsetBlock<ui32>>{std::move(block)})
                )
            );
        }
    } else {
        const ui32 partSize = objectsGrouping.GetObjectCount() / partCount;
        ui32 currentPartObjectEnd = 0;
        ui32 currentPartGroupEnd = 0;
        for (ui32 part = 0; part < partCount; ++part) {
            currentPartObjectEnd =
                oldCvStyle ?
                (objectCount * (part + 1) / partCount) :
                Min(currentPartObjectEnd + partSize, objectCount);
            const ui32 lastGroupIdx = (
                part + 1 == partCount ?
                (objectsGrouping.GetGroupCount() - 1) :
                objectsGrouping.GetGroupIdxForObject(currentPartObjectEnd - 1)
            );
            TSubsetBlock<ui32> block{{currentPartGroupEnd, lastGroupIdx + 1}, 0};
            const ui32 blockSize = block.GetSize();
            CB_ENSURE(blockSize > 0, "Not enough objects for splitting into requested amount of parts");
            result.push_back(
                TArraySubsetIndexing<ui32>(
                    TRangesSubset<ui32>(blockSize, TVector<TSubsetBlock<ui32>>{std::move(block)})
                )
            );
            currentPartGroupEnd = lastGroupIdx + 1;
            currentPartObjectEnd = objectsGrouping.GetGroup(lastGroupIdx).End;
        }
    }
    return result;
}

