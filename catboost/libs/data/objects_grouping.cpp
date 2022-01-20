#include "objects_grouping.h"

#include <catboost/libs/helpers/permutation.h>

#include <util/generic/overloaded.h>
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
    if (objectsGrouping->IsTrivial()) {
        return TObjectsGroupingSubset(
            ::std::holds_alternative<TFullSubset<ui32>>(groupsSubset) ?
                objectsGrouping : MakeIntrusive<TObjectsGrouping>(groupsSubset.Size()),
            std::move(groupsSubset),
            groupSubsetOrder
        );
    } else {
        TMaybe<TArraySubsetIndexing<ui32>> objectsSubset;
        TVector<TGroupBounds> subsetGroupBounds;

        if (std::holds_alternative<TFullSubset<ui32>>(groupsSubset)) {
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
        } else if (const auto* ranges = std::get_if<TRangesSubset<ui32>>(&groupsSubset)) {
            const auto& groupsSubsetBlocks = ranges->Blocks;

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
        } else if (const auto* groupsSubsetIndices = std::get_if<TIndexedSubset<ui32>>(&groupsSubset)) {
            auto nontrivialSrcGroups = objectsGrouping->GetNonTrivialGroups();

            subsetGroupBounds.reserve(groupsSubset.Size());

            TVector<TSubsetBlock<ui32>> objectsSubsetBlocks;
            objectsSubsetBlocks.reserve(groupsSubsetIndices->size());

            ui32 objectsDstBegin = 0;
            for (auto srcGroupIdx : *groupsSubsetIndices) {
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
        if (objectsSubset.IsFullSubset()) {
            Y_ASSERT(subsetOrder == EObjectsOrder::Ordered);
            return GetSubset(objectsGrouping, TArraySubsetIndexing<ui32>(TFullSubset<ui32>(nontrivialSrcGroups.size())), subsetOrder);;
        }
        ui32 subsetCurrentGroupOffset = 0;

        constexpr TStringBuf INVARIANT_MESSAGE =
            " subset groups invariant (if group is present in the subset all it's member objects"
            " must present and be in the same order within a group). This constraint might be"
            " relaxed in the future.";
        // TODO(kirillovs): get rid of N * log(N)
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

void NCB::TrainTestSplit(
    const TObjectsGrouping& objectsGrouping,
    double trainPart,
    TArraySubsetIndexing<ui32>* trainIndices,
    TArraySubsetIndexing<ui32>* testIndices
) {
    const ui32 objectCount = objectsGrouping.GetObjectCount();
    const ui32 trainSize = static_cast<ui32>(trainPart * objectCount);
    CB_ENSURE(trainSize > 0 && trainSize < objectCount, "Can't split with provided trainPart");
    TSubsetBlock<ui32> trainBlock;
    TSubsetBlock<ui32> testBlock;
    if (objectsGrouping.IsTrivial()) {
        trainBlock.SrcBegin = 0;
        trainBlock.SrcEnd = trainSize;
        testBlock.SrcBegin = trainBlock.SrcEnd;
        testBlock.SrcEnd = objectCount;
    } else {
        const ui32 lastGroupIdx = objectsGrouping.GetGroupIdxForObject(trainSize - 1);
        trainBlock = TSubsetBlock<ui32>{{0, lastGroupIdx + 1}, 0};
        CB_ENSURE(trainBlock.GetSize() > 0, "Not enough objects to give train split");
        testBlock = TSubsetBlock<ui32>{{lastGroupIdx + 1, objectsGrouping.GetGroupIdxForObject(objectCount - 1)}, 0};
        CB_ENSURE(testBlock.GetSize() > 0, "Not enough objects to give test split");
    }
    *trainIndices = TArraySubsetIndexing<ui32>(
        TRangesSubset<ui32>(trainBlock.GetSize(), TVector<TSubsetBlock<ui32>>{std::move(trainBlock)})
    );
    *testIndices = TArraySubsetIndexing<ui32>(
        TRangesSubset<ui32>(testBlock.GetSize(), TVector<TSubsetBlock<ui32>>{std::move(testBlock)})
    );
}
static void AlignBlockEndToGroupEnd(
    const TObjectsGrouping& objectsGrouping,
    TSubsetBlock<ui32>* block
) {
    if (objectsGrouping.IsTrivial() || block->GetSize() == 0) {
        return;
    }
    block->SrcEnd = objectsGrouping.GetGroup(objectsGrouping.GetGroupIdxForObject(block->SrcEnd - 1)).End;
}

static TSubsetBlock<ui32> GetGroupBlock(
    const TObjectsGrouping& objectsGrouping,
    const TSubsetBlock<ui32>& block
) {
    if (objectsGrouping.IsTrivial() || block.GetSize() == 0) {
        return block;
    }
    const auto srcBegin = objectsGrouping.GetGroupIdxForObject(block.SrcBegin);
    const auto srcEnd = objectsGrouping.GetGroupIdxForObject(block.SrcEnd - 1) + 1;
    return {{srcBegin, srcEnd}, /*DstBegin*/0};
}

TVector<TArraySubsetIndexing<ui32>> NCB::SplitByObjects(
    const TObjectsGrouping& objectsGrouping,
    ui32 partSizeInObjects
) {
    const ui32 objectCount = objectsGrouping.GetObjectCount();
    TVector<TArraySubsetIndexing<ui32>> result;
    TSubsetBlock<ui32> block{{/*SrcBegin*/0, /*SrcEnd*/Min(objectCount, partSizeInObjects)}, /*DstBegin*/0};
    while (block.GetSize() > 0) {
        AlignBlockEndToGroupEnd(objectsGrouping, &block);
        const auto groupBlock = GetGroupBlock(objectsGrouping, block);
        result.push_back(
            TArraySubsetIndexing<ui32>(
                TRangesSubset<ui32>(groupBlock.GetSize(), TVector<TSubsetBlock<ui32>>{groupBlock})
            )
        );
        block.SrcBegin = block.SrcEnd;
        block.SrcEnd = Min(objectCount, block.SrcBegin + partSizeInObjects);
    }
    return result;
}

TVector<TArraySubsetIndexing<ui32>> NCB::SplitByGroups(
    const TObjectsGrouping& objectsGrouping,
    ui32 partSizeInGroups
) {
    const ui32 groupCount = objectsGrouping.GetGroupCount();
    TVector<TArraySubsetIndexing<ui32>> result;
    TSubsetBlock<ui32> block{{/*SrcBegin*/0, /*SrcEnd*/Min(groupCount, partSizeInGroups)}, /*DstBegin*/0};
    while (block.GetSize() > 0) {
        result.push_back(
            TArraySubsetIndexing<ui32>(
                TRangesSubset<ui32>(block.GetSize(), TVector<TSubsetBlock<ui32>>{block})
            )
        );
        block.SrcBegin = block.SrcEnd;
        block.SrcEnd = Min(groupCount, block.SrcBegin + partSizeInGroups);
    }
    return result;
}

static TVector<TArraySubsetIndexing<ui32>> QuantileSplit(
    std::function<bool(ui32, ui32)> isLargeSubset,
    const TObjectsGrouping& objectsGrouping,
    TConstArrayRef<ui64> timestamps,
    ui64 timesplitQuantileTimestamp
) {
    TVector<TArraySubsetIndexing<ui32>> result;

    TIndexedSubset<ui32> learnSubset;
    ui32 learnSubsetSizeInObjects = 0;
    TIndexedSubset<ui32> testSubset;
    for (ui32 groupIdx : xrange(objectsGrouping.GetGroupCount())) {
        const auto group = objectsGrouping.GetGroup(groupIdx);
        const auto groupTimestamp = timestamps[group.Begin];
        if (groupTimestamp <= timesplitQuantileTimestamp) {
            if (isLargeSubset(learnSubset.size(), learnSubsetSizeInObjects)) {
                result.emplace_back(TIndexedSubset<ui32>(learnSubset));
                learnSubset.clear();
                learnSubsetSizeInObjects = 0;
            }
            learnSubset.push_back(groupIdx);
            learnSubsetSizeInObjects += group.GetSize();
        } else {
            testSubset.push_back(groupIdx);
        }
    }
    if (!learnSubset.empty()) {
        result.emplace_back(TIndexedSubset<ui32>(learnSubset));
    }
    CB_ENSURE(testSubset.size() > 0, "No groups left for test. Please decrease timesplit quantile.");
    result.emplace_back(TIndexedSubset<ui32>(testSubset));
    return result;
}

// Returns vector {learn_fold, ..., learn_fold, test_fold}
// Sum of sizes of all learn folds is sum of sizes of all groups having timestamp <= timesplitQuantileTimestamp
// Size of each learn fold is learnPartSizeInObjects objects rounded upward to group bound
// Size of test_fold is (object count - sum of sizes of all learn folds)
TVector<TArraySubsetIndexing<ui32>> NCB::QuantileSplitByObjects(
    const TObjectsGrouping& objectsGrouping,
    TConstArrayRef<ui64> timestamps,
    ui64 timesplitQuantileTimestamp,
    ui32 learnPartSizeInObjects
) {
    CB_ENSURE(learnPartSizeInObjects > 0, "Fold size must be positive");
    const auto isSubsetLarge = [learnPartSizeInObjects] (ui32 /*groupCount*/, ui32 objectCount) {
        return objectCount >= learnPartSizeInObjects;
    };
    return QuantileSplit(isSubsetLarge, objectsGrouping, timestamps, timesplitQuantileTimestamp);
}


// Returns vector {learn_fold, ..., learn_fold, test_fold}
// Sum of sizes of all learn folds is number of all groups having timestamp <= timesplitQuantileTimestamp
// Size of each learn fold is learnPartSizeInGroups groups
// Size of test_fold is (group count - sum of sizes of all learn folds)
TVector<TArraySubsetIndexing<ui32>> NCB::QuantileSplitByGroups(
    const TObjectsGrouping& objectsGrouping,
    TConstArrayRef<ui64> timestamps,
    ui64 timesplitQuantileTimestamp,
    ui32 learnPartSizeInGroups
) {
    CB_ENSURE(learnPartSizeInGroups > 0, "Fold size must be positive");
    const auto isSubsetLarge = [learnPartSizeInGroups] (ui32 groupCount, ui32 /*objectCount*/) {
        return groupCount >= learnPartSizeInGroups;
    };
    return QuantileSplit(isSubsetLarge, objectsGrouping, timestamps, timesplitQuantileTimestamp);
}

TTimeSeriesTrainTestSubsets NCB::TimeSeriesSplit(
    const TObjectsGrouping& objectsGrouping,
    ui32 partCount,
    bool oldCvStyle
) {
    const auto regularTestSubsets = Split(
        objectsGrouping,
        partCount + 1,
        oldCvStyle
    );
    TVector<TArraySubsetIndexing<ui32>> trainSubsets;
    TVector<TArraySubsetIndexing<ui32>> testSubsets;

    for (ui32 part = 1; part < partCount + 1; ++part) {
        TRangesSubset<ui32> currentEndBlocks = regularTestSubsets[part - 1].Get<TRangesSubset<ui32>>();
        TSubsetBlock<ui32> blockTrain;
        blockTrain.SrcBegin = 0;
        blockTrain.SrcEnd = currentEndBlocks.Blocks[0].SrcEnd;
        blockTrain.DstBegin = 0;
        trainSubsets.push_back(
            TArraySubsetIndexing<ui32>(
                TRangesSubset<ui32>(blockTrain.GetSize(), TVector<TSubsetBlock<ui32>>{std::move(blockTrain)})
            )
        );

        testSubsets.emplace_back(regularTestSubsets[part]);
    }

    return std::make_pair(trainSubsets, testSubsets);
}
