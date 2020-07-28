#include <catboost/libs/data/objects_grouping.h>

#include <catboost/private/libs/index_range/index_range.h>
#include <catboost/libs/helpers/vector_helpers.h>

#include <util/generic/xrange.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(TObjectsGrouping) {
    Y_UNIT_TEST(Basic) {
        // trivial
        {
            TObjectsGrouping grouping(10);

            UNIT_ASSERT_VALUES_EQUAL(grouping.GetObjectCount(), 10);
            UNIT_ASSERT_VALUES_EQUAL(grouping.GetGroupCount(), 10);
            UNIT_ASSERT(grouping.IsTrivial());

            for (auto i : xrange(10)) {
                UNIT_ASSERT_EQUAL(grouping.GetGroup(i), TGroupBounds(i, i + 1));
            }

            UNIT_ASSERT_EXCEPTION(grouping.GetNonTrivialGroups(), TCatBoostException);
        }

        // non-trivial
        {
            TVector<TGroupBounds> groupsBounds = {{0, 1}, {1, 3}, {3, 10}};

            TObjectsGrouping grouping{TVector<TGroupBounds>(groupsBounds)};

            UNIT_ASSERT_VALUES_EQUAL(grouping.GetObjectCount(), 10);
            UNIT_ASSERT_VALUES_EQUAL(grouping.GetGroupCount(), (ui32)groupsBounds.size());
            UNIT_ASSERT(!grouping.IsTrivial());

            for (auto i : xrange(groupsBounds.size())) {
                UNIT_ASSERT_EQUAL(grouping.GetGroup(i), groupsBounds[i]);
            }

            UNIT_ASSERT(Equal(grouping.GetNonTrivialGroups(), groupsBounds));
        }

        // bad
        {
            TVector<TGroupBounds> groupsBounds = {{0, 1}, {2, 5}};

            UNIT_ASSERT_EXCEPTION(TObjectsGrouping(std::move(groupsBounds)), TCatBoostException);
        }
    }

    Y_UNIT_TEST(GetSubset) {
        TVector<TGroupBounds> groups1 = {
            {0, 1},
            {1, 3},
            {3, 10},
            {10, 12},
            {12, 13},
            {13, 20},
            {20, 25},
            {25, 27},
            {27, 33},
            {33, 42}
        };

        TVector<TObjectsGrouping> objectGroupings;
        objectGroupings.emplace_back(10); // trivial
        objectGroupings.emplace_back(TVector<TGroupBounds>(groups1));


        TVector<TArraySubsetIndexing<ui32>> subsetVector;
        TVector<EObjectsOrder> subsetGroupOrders;

        subsetVector.emplace_back(TFullSubset<ui32>(10));
        subsetGroupOrders.emplace_back(EObjectsOrder::Ordered);

        {
            TVector<TIndexRange<ui32>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));
            subsetVector.emplace_back(TRangesSubset<ui32>(savedIndexRanges));
            subsetGroupOrders.emplace_back(EObjectsOrder::Undefined);
        }

        subsetVector.emplace_back(TIndexedSubset<ui32>{8, 9, 0, 2});
        subsetGroupOrders.emplace_back(EObjectsOrder::Undefined);


        using TExpectedMapIndex = std::pair<ui32, ui32>;

        // (objectGroupings idx, subsetVector idx) -> expectedSubset
        THashMap<TExpectedMapIndex, TObjectsGroupingSubset> expectedSubsets;

        for (auto subsetIdx : xrange(subsetVector.size())) {
            expectedSubsets.emplace(
                TExpectedMapIndex(0, subsetIdx),
                TObjectsGroupingSubset(
                    MakeIntrusive<TObjectsGrouping>(subsetVector[subsetIdx].Size()),
                    TArraySubsetIndexing<ui32>(subsetVector[subsetIdx]),
                    subsetGroupOrders[subsetIdx]
                )
            );
        }

        {
            TSavedIndexRanges<ui32> savedIndexRanges{TVector<TIndexRange<ui32>>(groups1)};
            expectedSubsets.emplace(
                TExpectedMapIndex(1, 0),
                TObjectsGroupingSubset(
                    MakeIntrusive<TObjectsGrouping>(TVector<TGroupBounds>(groups1)),
                    TArraySubsetIndexing<ui32>(subsetVector[0]),
                    subsetGroupOrders[0],
                    MakeMaybe<TArraySubsetIndexing<ui32>>(TFullSubset<ui32>(42)),
                    subsetGroupOrders[0]
                )
            );
        }

        {
            TVector<TIndexRange<ui32>> indexRanges{{25, 42}, {3, 10}, {12, 20}};
            TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));
            expectedSubsets.emplace(
                TExpectedMapIndex(1, 1),
                TObjectsGroupingSubset(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 2}, {2, 8}, {8, 17}, {17, 24}, {24, 25}, {25, 32}}
                    ),
                    TArraySubsetIndexing<ui32>(subsetVector[1]),
                    subsetGroupOrders[1],
                    MakeMaybe<TArraySubsetIndexing<ui32>>(TRangesSubset<ui32>(savedIndexRanges)),
                    subsetGroupOrders[1]
                )
            );
        }

        {
            TVector<TIndexRange<ui32>> indexRanges{{27, 33}, {33, 42}, {0, 1}, {3, 10}};
            TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));
            expectedSubsets.emplace(
                TExpectedMapIndex(1, 2),
                TObjectsGroupingSubset(
                    MakeIntrusive<TObjectsGrouping>(
                        TVector<TGroupBounds>{{0, 6}, {6, 15}, {15, 16}, {16, 23}}
                    ),
                    TArraySubsetIndexing<ui32>(subsetVector[2]),
                    subsetGroupOrders[2],
                    MakeMaybe<TArraySubsetIndexing<ui32>>(TRangesSubset<ui32>(savedIndexRanges)),
                    subsetGroupOrders[2]
                )
            );
        }

        for (auto objectGroupingIdx : xrange(objectGroupings.size())) {
            for (auto subsetIdx : xrange(subsetVector.size())) {
                TObjectsGroupingSubset subset = GetSubset(
                    MakeIntrusive<TObjectsGrouping>(objectGroupings[objectGroupingIdx]),
                    TArraySubsetIndexing<ui32>(subsetVector[subsetIdx]),
                    subsetGroupOrders[subsetIdx]
                );
                const auto& expectedSubset = expectedSubsets.at(TExpectedMapIndex(objectGroupingIdx, subsetIdx));
                UNIT_ASSERT_EQUAL(subset, expectedSubset);
            }
        }

    }

    Y_UNIT_TEST(GetGroupingSubsetFromObjectsSubset) {
        {
             auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(ui32(10));
             auto objectsSubset = TArraySubsetIndexing<ui32>(TFullSubset<ui32>(10));

             auto objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
                 objectsGrouping,
                 std::move(objectsSubset),
                 EObjectsOrder::Undefined
             );

             UNIT_ASSERT(
                 IndicesEqual(
                     objectsGroupingSubset.GetGroupsIndexing(),
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
                 )
             );
             UNIT_ASSERT(
                 IndicesEqual(
                     objectsGroupingSubset.GetObjectsIndexing(),
                     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
                 )
             );
        }
        {
             auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(ui32(10));
             auto objectsSubset = TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>{3, 7, 1});

             auto objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
                 objectsGrouping,
                 std::move(objectsSubset),
                 EObjectsOrder::Undefined
             );

             UNIT_ASSERT(
                 IndicesEqual(
                     objectsGroupingSubset.GetGroupsIndexing(),
                     {3, 7, 1}
                 )
             );
             UNIT_ASSERT(
                 IndicesEqual(
                     objectsGroupingSubset.GetObjectsIndexing(),
                     {3, 7, 1}
                 )
             );
        }
        {
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(
                TVector<TGroupBounds>{{0, 1}, {1, 3}, {3, 6}, {6, 7}, {7, 9}}
            );
            auto objectsSubset = TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>{3, 4, 5, 0});

            auto objectsGroupingSubset = GetGroupingSubsetFromObjectsSubset(
                objectsGrouping,
                std::move(objectsSubset),
                EObjectsOrder::Undefined
            );

            UNIT_ASSERT(
                IndicesEqual(
                    objectsGroupingSubset.GetGroupsIndexing(),
                    {2, 0}
                )
            );
            UNIT_ASSERT(
                IndicesEqual(
                    objectsGroupingSubset.GetObjectsIndexing(),
                    {3, 4, 5, 0}
                )
            );
        }
        {
            // partial groups case
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(
                TVector<TGroupBounds>{{0, 1}, {1, 3}, {3, 6}, {6, 7}, {7, 9}}
            );
            auto objectsSubset = TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>{3, 4});

            UNIT_ASSERT_EXCEPTION(
                GetGroupingSubsetFromObjectsSubset(
                    objectsGrouping,
                    std::move(objectsSubset),
                    EObjectsOrder::Undefined
                ),
                TCatBoostException
            );
        }
        {
            // order violation case
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(
                TVector<TGroupBounds>{{0, 1}, {1, 3}, {3, 6}, {6, 7}, {7, 9}}
            );
            auto objectsSubset = TArraySubsetIndexing<ui32>(TIndexedSubset<ui32>{7, 8, 0});

            UNIT_ASSERT_EXCEPTION(
                GetGroupingSubsetFromObjectsSubset(
                    objectsGrouping,
                    std::move(objectsSubset),
                    EObjectsOrder::Ordered
                ),
                TCatBoostException
            );
        }
    }

    Y_UNIT_TEST(GetGroupIdxForObject) {
        // trivial
        {
            TObjectsGrouping grouping(10);

            for (auto i : xrange(10)) {
                UNIT_ASSERT_VALUES_EQUAL(grouping.GetGroupIdxForObject(i), i);
            }

            UNIT_ASSERT_EXCEPTION(grouping.GetGroupIdxForObject(10), TCatBoostException);
            UNIT_ASSERT_EXCEPTION(grouping.GetGroupIdxForObject(100), TCatBoostException);
        }

        // non-trivial
        {
            TVector<TGroupBounds> groupsBounds = {{0, 1}, {1, 3}, {3, 10}, {10, 17}, {17, 22}};

            TObjectsGrouping grouping{TVector<TGroupBounds>(groupsBounds)};

            // objectIdx, expectedGroupIds
            TVector<std::pair<ui32, ui32>> expectedObjectToGroupIdxs = {
                {0, 0}, {1, 1}, {2, 1}, {3, 2}, {5, 2}, {8, 2}, {9, 2}, {10, 3}, {11, 3}, {17, 4}, {21, 4}
            };

            for (auto expectedObjectToGroupIdx : expectedObjectToGroupIdxs) {
                UNIT_ASSERT_VALUES_EQUAL(
                    grouping.GetGroupIdxForObject(expectedObjectToGroupIdx.first),
                    expectedObjectToGroupIdx.second
                );
            }

            UNIT_ASSERT_EXCEPTION(grouping.GetGroupIdxForObject(22), TCatBoostException);
            UNIT_ASSERT_EXCEPTION(grouping.GetGroupIdxForObject(100), TCatBoostException);
        }
    }

    Y_UNIT_TEST(Shuffle) {
        {
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(ui32(10));
            ui32 permuteBlockSize = 1;

            TRestorableFastRng64 rand(0);
            auto shuffledObjectsGroupingSubset = Shuffle(objectsGrouping, permuteBlockSize, &rand);

            UNIT_ASSERT(
                IndicesEqual(
                    shuffledObjectsGroupingSubset.GetGroupsIndexing(),
                    {8, 6, 5, 0, 3, 1, 7, 4, 9, 2}
                )
            );
            UNIT_ASSERT(
                IndicesEqual(
                    shuffledObjectsGroupingSubset.GetObjectsIndexing(),
                    {8, 6, 5, 0, 3, 1, 7, 4, 9, 2}
                )
            );
        }
        {
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(ui32(13));
            ui32 permuteBlockSize = 4;

            TRestorableFastRng64 rand(0);
            auto shuffledObjectsGroupingSubset = Shuffle(objectsGrouping, permuteBlockSize, &rand);

            UNIT_ASSERT(
                IndicesEqual(
                    shuffledObjectsGroupingSubset.GetGroupsIndexing(),
                    {8, 9, 10, 11, 12, 4, 5, 6, 7, 0, 1, 2, 3}
                )
            );
            UNIT_ASSERT(
                IndicesEqual(
                    shuffledObjectsGroupingSubset.GetObjectsIndexing(),
                    {8, 9, 10, 11, 12, 4, 5, 6, 7, 0, 1, 2, 3}
                )
            );
        }
        {
            auto objectsGrouping = MakeIntrusive<TObjectsGrouping>(
                TVector<TGroupBounds>{{0, 2}, {2, 5}, {5, 8}, {8, 9}, {9, 10}, {10, 12}}
            );
            ui32 permuteBlockSize = 1;

            TRestorableFastRng64 rand(0);
            auto shuffledObjectsGroupingSubset = Shuffle(objectsGrouping, permuteBlockSize, &rand);

            UNIT_ASSERT(
                IndicesEqual(shuffledObjectsGroupingSubset.GetGroupsIndexing(), { 2, 4, 5, 0, 3, 1 })
            );
            UNIT_ASSERT(
                IndicesEqual(
                    shuffledObjectsGroupingSubset.GetObjectsIndexing(),
                    {7, 5, 6, 9, 11, 10, 1, 0, 8, 2, 4, 3}
                )
            );
        }
    }

    Y_UNIT_TEST(TimeSeriesSplit) {
        const ui32 objectCount = 8;
        TObjectsGrouping objectsGrouping(objectCount);
        const ui32 foldCount = 3;
        const bool oldStyle = false;
        const auto result = TimeSeriesSplit(objectsGrouping, foldCount, oldStyle);

        TVector<TArraySubsetIndexing<ui32>> expectedTrainIndices;
        TVector<TArraySubsetIndexing<ui32>> expectedTestIndices;

        auto getRangesSubset = [](ui32 begin, ui32 end) {
            TSubsetBlock<ui32> blockBuffer;
            blockBuffer.DstBegin = 0;
            blockBuffer.SrcBegin = begin;
            blockBuffer.SrcEnd = end;
            return TRangesSubset<ui32>(blockBuffer.GetSize(), TVector<TSubsetBlock<ui32>>{blockBuffer});
        };

        expectedTrainIndices.emplace_back(getRangesSubset(0, 2));
        expectedTrainIndices.emplace_back(getRangesSubset(0, 4));
        expectedTrainIndices.emplace_back(getRangesSubset(0, 6));

        expectedTestIndices.emplace_back(getRangesSubset(2, 4));
        expectedTestIndices.emplace_back(getRangesSubset(4, 6));
        expectedTestIndices.emplace_back(getRangesSubset(6, 8));

        UNIT_ASSERT_EQUAL(result.first, expectedTrainIndices);
        UNIT_ASSERT_EQUAL(result.second, expectedTestIndices);
    }

    Y_UNIT_TEST(QuantileSplitObjects) {
        const ui32 objectCount = 8;
        const TObjectsGrouping objectsGrouping(objectCount);
        const TVector<ui64> timestamps{0, 1, 2, 3, 4, 5, 6, 7};
        const auto timesplitQuantileTimestamp = timestamps[5];
        const ui32 learnPartSizeInObjects = 3;
        const auto result = QuantileSplitByObjects(objectsGrouping, timestamps, timesplitQuantileTimestamp, learnPartSizeInObjects);
        TVector<TArraySubsetIndexing<ui32>> learnTestIndices;
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({0, 1, 2}));
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({3, 4, 5}));
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({6, 7}));
        UNIT_ASSERT_EQUAL(result, learnTestIndices);
    }

    Y_UNIT_TEST(QuantileSplitGroups) {
        const TObjectsGrouping objectsGrouping(TVector<TGroupBounds>({TGroupBounds(0, 2), TGroupBounds(2, 5), TGroupBounds(5, 8)}));
        const TVector<ui64> timestamps{0, 0, 2, 2, 2, 6, 6, 6};
        const auto timesplitQuantileTimestamp = timestamps[2];
        const ui32 learnPartSizeInGroups = 1;
        const auto result = QuantileSplitByGroups(objectsGrouping, timestamps, timesplitQuantileTimestamp, learnPartSizeInGroups);
        TVector<TArraySubsetIndexing<ui32>> learnTestIndices;
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({0}));
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({1}));
        learnTestIndices.emplace_back(TIndexedSubset<ui32>({2}));
        UNIT_ASSERT_EQUAL(result, learnTestIndices);
    }
}
