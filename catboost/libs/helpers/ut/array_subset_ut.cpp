#include <catboost/libs/helpers/array_subset.h>

#include <util/datetime/base.h>
#include <util/generic/is_in.h>
#include <util/generic/hash.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>

#include <utility>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(TArraySubset) {
    Y_UNIT_TEST(TestNullArguments) {
        UNIT_ASSERT_EXCEPTION(
            ([]{
                NCB::TArraySubset<TVector<int>> arraySubset{nullptr, nullptr};
            }()),
            TCatBoostException
        );
        UNIT_ASSERT_EXCEPTION(
            ([]{
                TVector<int> v(1, 0);
                NCB::TArraySubset<TVector<int>> arraySubset{&v, nullptr};
            }()),
            TCatBoostException
        );
        UNIT_ASSERT_EXCEPTION(
            ([]{
                NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{0} );
                NCB::TArraySubset<TVector<int>> arraySubset{nullptr, &arraySubsetIndexing};
            }()),
            TCatBoostException
        );
    }

    Y_UNIT_TEST(TestBadBlockSize) {
        UNIT_ASSERT_EXCEPTION(
            ([]{
                TVector<int> v(1, 0);
                NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{1} );
                NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

                NPar::TLocalExecutor localExecutor;

                arraySubset.ParallelForEach(
                    [](size_t /*index*/, int /*value*/) { Sleep(TDuration::MilliSeconds(1)); },
                    &localExecutor,
                    0
                );
            }()),
            TCatBoostException
        );
    }

    void TestOneCase(
        const NCB::TArraySubset<TVector<int>>& arraySubset,
        const TVector<int>& expectedSubset
    ) {
        UNIT_ASSERT_EQUAL(arraySubset.Size(), expectedSubset.size());

        // ForEach
        {
            size_t expectedIndex = 0;
            arraySubset.ForEach([&](size_t index, int value) {
                UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
                UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);
                ++expectedIndex;
            });
            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, expectedSubset.size());
        }

        // ParallelForEach
        {
            NPar::TLocalExecutor localExecutor;
            localExecutor.RunAdditionalThreads(3);

            for (size_t approximateBlockSize : xrange(0, 12)) { // use 0 as undefined
                TVector<bool> indicesIterated(expectedSubset.size(), false);
                arraySubset.ParallelForEach(
                    [&](size_t index, int value) {
                        UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);

                        auto& indexIterated = indicesIterated.at(index);
                        UNIT_ASSERT(!indexIterated); // each index must be visited only once
                        indexIterated = true;
                    },
                    &localExecutor,
                    approximateBlockSize != 0 ? TMaybe<size_t>(approximateBlockSize) : Nothing()
                );
                UNIT_ASSERT(!IsIn(indicesIterated, false)); // each index was visited
            }
        }

        // external iteration
        {
            auto* src = arraySubset.GetSrc();
            const auto* subsetIndexing = arraySubset.GetSubsetIndexing();

            for (size_t approximateBlockSize : xrange(1, 12)) {
                const NCB::TSimpleIndexRangesGenerator<size_t> parallelUnitRanges =
                    subsetIndexing->GetParallelUnitRanges(approximateBlockSize);

                TVector<bool> indicesIterated(expectedSubset.size(), false);

                for (size_t unitRangeIdx : xrange(parallelUnitRanges.RangesCount())) {
                    auto unitRange = parallelUnitRanges.GetRange(unitRangeIdx);
                    auto elementRange = subsetIndexing->GetElementRangeFromUnitRange(unitRange);

                    size_t expectedIndex = elementRange.Begin;
                    subsetIndexing->ForEachInSubRange(
                        unitRange,
                        [&](size_t index, size_t srcIndex) {
                            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
                            ++expectedIndex;

                            UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], (*src)[srcIndex]);

                            auto& indexIterated = indicesIterated.at(index);
                            UNIT_ASSERT(!indexIterated); // each index must be visited only once
                            indexIterated = true;
                        }
                    );
                    UNIT_ASSERT_VALUES_EQUAL(expectedIndex, elementRange.End);
                }
                UNIT_ASSERT(!IsIn(indicesIterated, false)); // each index was visited
            }
        }

        // test Equal
        UNIT_ASSERT(Equal<int>(expectedSubset, arraySubset));

        UNIT_ASSERT(!Equal(TConstArrayRef<int>(), arraySubset));

        {
            TVector<int> modifiedExpectedSubset = expectedSubset;
            ++modifiedExpectedSubset.back();

            UNIT_ASSERT(!Equal<int>(modifiedExpectedSubset, arraySubset));
        }
        {
            TVector<int> modifiedExpectedSubset = expectedSubset;
            modifiedExpectedSubset.push_back(11);

            UNIT_ASSERT(!Equal<int>(modifiedExpectedSubset, arraySubset));
        }
    }

    enum class EIterationType {
        ForEach,
        ParallelForEach,
        External
    };

    template <class F>
    void TestMutable(
        const TVector<int>& array,
        const NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing,
        const TVector<int>& expectedModifiedSubset,
        F&& f
    ) {
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Size(), expectedModifiedSubset.size());

        for (auto iterationType : {
                EIterationType::ForEach,
                EIterationType::ParallelForEach,
                EIterationType::External
             })
        {
            TVector<int> arrayCopy = array;
            NCB::TArraySubset<TVector<int>> arraySubset{&arrayCopy, &arraySubsetIndexing};

            switch (iterationType) {
                case EIterationType::ForEach:
                    arraySubset.ForEach(f);
                    break;
                case EIterationType::ParallelForEach: {
                        NPar::TLocalExecutor localExecutor;
                        localExecutor.RunAdditionalThreads(3);

                        arraySubset.ParallelForEach(f, &localExecutor, 3);
                    }
                    break;
                case EIterationType::External: {
                    const auto* subsetIndexing = arraySubset.GetSubsetIndexing();

                    const NCB::TSimpleIndexRangesGenerator<size_t> parallelUnitRanges =
                        subsetIndexing->GetParallelUnitRanges(3);

                    for (size_t unitRangeIdx : xrange(parallelUnitRanges.RangesCount())) {
                        auto unitRange = parallelUnitRanges.GetRange(unitRangeIdx);
                        subsetIndexing->ForEachInSubRange(
                            unitRange,
                            [&](size_t index, size_t srcIndex) {
                                f(index, arrayCopy[srcIndex]);
                            }
                        );
                    }
                }
            }
        }
    }

    void TestGetSubset(
        const TVector<int>& v,
        const NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing,
        const TVector<int>& expectedVSubset,
        TMaybe<NPar::ILocalExecutor*> localExecutor
    ) {
        {
            TVector<int> vSubset = NCB::GetSubset<int>(v, arraySubsetIndexing, localExecutor);
            UNIT_ASSERT_VALUES_EQUAL(vSubset, expectedVSubset);
        }
        {
            TMaybe<TVector<int>> vSubset = NCB::GetSubsetOfMaybeEmpty<int>(
                MakeMaybe((TConstArrayRef<int>)v),
                arraySubsetIndexing,
                localExecutor
            );
            UNIT_ASSERT(vSubset);
            UNIT_ASSERT_EQUAL(*vSubset, expectedVSubset);
        }
        {
            TMaybe<TVector<int>> vSubset = NCB::GetSubsetOfMaybeEmpty<int>(
                TMaybe<TConstArrayRef<int>>(),
                arraySubsetIndexing,
                localExecutor
            );
            UNIT_ASSERT_EQUAL(vSubset, Nothing());
        }
    }

    void TestGetSubset(
        const TVector<int>& v,
        const NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing,
        const TVector<int>& expectedVSubset
    ) {
        TestGetSubset(v, arraySubsetIndexing, expectedVSubset, Nothing());

        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(3);
        TestGetSubset(v, arraySubsetIndexing, expectedVSubset, &localExecutor);
    }

    Y_UNIT_TEST(TestFullSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{v.size()} );

        NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

        TestOneCase(arraySubset, v);

        TVector<int> expectedModifiedSubset = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

        TestMutable(
            v,
            arraySubsetIndexing,
            expectedModifiedSubset,
            [] (size_t /*idx*/, int& value) {
                value += 5;
            }
        );

        UNIT_ASSERT(arraySubset.Find([](size_t /*idx*/, int value) { return value == 15; }));
        UNIT_ASSERT(!arraySubset.Find([](size_t /*idx*/, int value) { return value == 0; }));

        TestGetSubset(v, arraySubsetIndexing, v, Nothing());
    }

    Y_UNIT_TEST(TestRangesSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        TVector<NCB::TIndexRange<size_t>> indexRanges{{7, 10}, {2, 3}, {4, 6}};

        NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( (NCB::TRangesSubset<size_t>(savedIndexRanges)) );
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Get<NCB::TRangesSubset<size_t>>().Size, 6);

        TVector<int> expectedSubset = {17, 18, 19, 12, 14, 15};
        NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

        TestOneCase(arraySubset, expectedSubset);

        TVector<int> expectedModifiedSubset = {22, 23, 24, 17, 19, 20};

        TestMutable(
            v,
            arraySubsetIndexing,
            expectedModifiedSubset,
            [] (size_t /*idx*/, int& value) {
                value += 5;
            }
        );

        UNIT_ASSERT(arraySubset.Find([](size_t /*idx*/, int value) { return value == 19; }));
        UNIT_ASSERT(!arraySubset.Find([](size_t /*idx*/, int value) { return value == 11; }));

        TestGetSubset(v, arraySubsetIndexing, expectedSubset);
    }


    Y_UNIT_TEST(TestIndexedSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1} );
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Get<NCB::TIndexedSubset<size_t>>().size(), 5);

        TVector<int> expectedSubset = {16, 15, 12, 10, 11};
        NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

        TestOneCase(arraySubset, expectedSubset);

        TVector<int> expectedModifiedSubset = {21, 20, 17, 15, 16};

        TestMutable(
            v,
            arraySubsetIndexing,
            expectedModifiedSubset,
            [] (size_t /*idx*/, int& value) {
                value += 5;
            }
        );

        UNIT_ASSERT(arraySubset.Find([](size_t /*idx*/, int value) { return value == 16; }));
        UNIT_ASSERT(!arraySubset.Find([](size_t /*idx*/, int value) { return value == 17; }));

        TestGetSubset(v, arraySubsetIndexing, expectedSubset);
    }

    Y_UNIT_TEST(TestGetConsecutiveSubsetBegin) {
        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>( NCB::TFullSubset<size_t>(0) ).GetConsecutiveSubsetBegin(),
            TMaybe<size_t>(0)
        );
        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>( NCB::TFullSubset<size_t>(10) ).GetConsecutiveSubsetBegin(),
            TMaybe<size_t>(0)
        );

        {
            TVector<NCB::TIndexRange<size_t>> indexRanges{};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            UNIT_ASSERT_VALUES_EQUAL(
                NCB::TArraySubsetIndexing<size_t>(
                    NCB::TRangesSubset<size_t>(savedIndexRanges)
                ).GetConsecutiveSubsetBegin(),
                TMaybe<size_t>(0)
            );
        }
        {
            TVector<NCB::TIndexRange<size_t>> indexRanges{{3, 12}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            UNIT_ASSERT_VALUES_EQUAL(
                NCB::TArraySubsetIndexing<size_t>(
                    NCB::TRangesSubset<size_t>(savedIndexRanges)
                ).GetConsecutiveSubsetBegin(),
                TMaybe<size_t>(3)
            );
        }
        {
            TVector<NCB::TIndexRange<size_t>> indexRanges{{7, 10}, {10, 10}, {10, 20}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            UNIT_ASSERT_VALUES_EQUAL(
                NCB::TArraySubsetIndexing<size_t>(
                    NCB::TRangesSubset<size_t>(savedIndexRanges)
                ).GetConsecutiveSubsetBegin(),
                TMaybe<size_t>(7)
            );
        }
        {
            TVector<NCB::TIndexRange<size_t>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            UNIT_ASSERT_VALUES_EQUAL(
                NCB::TArraySubsetIndexing<size_t>(
                    NCB::TRangesSubset<size_t>(savedIndexRanges)
                ).GetConsecutiveSubsetBegin(),
                Nothing()
            );
        }

        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>( NCB::TIndexedSubset<size_t>{} ).GetConsecutiveSubsetBegin(),
            TMaybe<size_t>(0)
        );
        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>( NCB::TIndexedSubset<size_t>{10} ).GetConsecutiveSubsetBegin(),
            TMaybe<size_t>(10)
        );
        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>(
                NCB::TIndexedSubset<size_t>{8, 9, 10, 11, 12}
            ).GetConsecutiveSubsetBegin(),
            TMaybe<size_t>(8)
        );
        UNIT_ASSERT_VALUES_EQUAL(
            NCB::TArraySubsetIndexing<size_t>(
                NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1}
            ).GetConsecutiveSubsetBegin(),
            Nothing()
        );
    }

    Y_UNIT_TEST(TestCompose) {
        TVector<NCB::TArraySubsetIndexing<size_t>> srcs;
        {
            srcs.emplace_back( NCB::TFullSubset<size_t>(6) );

            TVector<NCB::TIndexRange<size_t>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            srcs.emplace_back( NCB::TRangesSubset<size_t>(savedIndexRanges) );

            srcs.emplace_back( NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1, 7} );
        }

        TVector<NCB::TArraySubsetIndexing<size_t>> srcSubsets;
        {
            srcSubsets.emplace_back( NCB::TFullSubset<size_t>(6) );

            TVector<NCB::TIndexRange<size_t>> indexRanges{{2, 3}, {3, 6}, {0, 2}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            srcSubsets.emplace_back( NCB::TRangesSubset<size_t>(savedIndexRanges) );

            srcSubsets.emplace_back( NCB::TIndexedSubset<size_t>{5, 2, 0, 1} );
        }


        using TExpectedMapIndex = std::pair<size_t, size_t>;

        // (src index, srcSubset index) -> expectedResult
        THashMap<TExpectedMapIndex, NCB::TArraySubsetIndexing<size_t>> expectedResults;

        for (auto srcIndex : xrange(srcs.size())) {
            expectedResults.emplace(TExpectedMapIndex(srcIndex, 0), srcs[srcIndex]);
        }

        for (auto srcSubsetIndex : xrange<size_t>(1, srcSubsets.size())) {
            expectedResults.emplace(TExpectedMapIndex(0, srcSubsetIndex), srcSubsets[srcSubsetIndex]);
        }

        {
            TVector<NCB::TIndexRange<size_t>> indexRanges{{9, 10}, {2, 3}, {4, 6}, {7, 9}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            expectedResults.emplace(
                TExpectedMapIndex(1, 1),
                NCB::TArraySubsetIndexing<size_t>( NCB::TRangesSubset<size_t>(savedIndexRanges) )
            );
        }

        expectedResults.emplace(
            TExpectedMapIndex(1, 2),
            NCB::TArraySubsetIndexing<size_t>( NCB::TIndexedSubset<size_t>{5, 9, 7, 8} )
        );

        expectedResults.emplace(
            TExpectedMapIndex(2, 1),
            NCB::TArraySubsetIndexing<size_t>( NCB::TIndexedSubset<size_t>{2, 0, 1, 7, 6, 5} )
        );

        expectedResults.emplace(
            TExpectedMapIndex(2, 2),
            NCB::TArraySubsetIndexing<size_t>( NCB::TIndexedSubset<size_t>{7, 2, 6, 5} )
        );

        for (auto srcIdx : xrange(srcs.size())) {
            for (auto srcSubsetIdx : xrange(srcSubsets.size())) {
                // result created as a named variable to simplify debugging
                auto result = Compose(srcs[srcIdx], srcSubsets[srcSubsetIdx]);
                UNIT_ASSERT_EQUAL(result, expectedResults.at(TExpectedMapIndex(srcIdx, srcSubsetIdx)));
            }
        }
    }

    Y_UNIT_TEST(TestBadCompose) {
        TVector<NCB::TArraySubsetIndexing<size_t>> srcs;
        {
            srcs.emplace_back( NCB::TFullSubset<size_t>(6) );

            TVector<NCB::TIndexRange<size_t>> indexRanges{{7, 10}, {2, 3}, {4, 6}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            srcs.emplace_back( NCB::TRangesSubset<size_t>(savedIndexRanges) );

            srcs.emplace_back( NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1, 7} );
        }

        TVector<NCB::TArraySubsetIndexing<size_t>> badSrcSubsets;
        {
            badSrcSubsets.emplace_back( NCB::TFullSubset<size_t>(4) );

            TVector<NCB::TIndexRange<size_t>> indexRanges{{2, 3}, {3, 8}, {0, 2}};
            NCB::TSavedIndexRanges<size_t> savedIndexRanges(std::move(indexRanges));
            badSrcSubsets.emplace_back( NCB::TRangesSubset<size_t>(savedIndexRanges) );

            badSrcSubsets.emplace_back( NCB::TIndexedSubset<size_t>{5, 2, 7, 1} );
        }

        for (auto srcIdx : xrange(srcs.size())) {
            for (auto badSrcSubsetIdx : xrange(badSrcSubsets.size())) {
                UNIT_ASSERT_EXCEPTION(
                    ([&]{
                        Compose(srcs[srcIdx], badSrcSubsets[badSrcSubsetIdx]);
                    }()),
                    TCatBoostException
                );
            }
        }
    }

    Y_UNIT_TEST(TestGetInvertedIndexing) {
        NPar::TLocalExecutor localExecutor;
        localExecutor.RunAdditionalThreads(2);

        {
            NCB::TArraySubsetIndexing<ui32> subset(NCB::TFullSubset<ui32>(4));
            NCB::TArraySubsetInvertedIndexing<ui32> invertedSubset
                = NCB::GetInvertedIndexing(subset, ui32(4), &localExecutor);
            UNIT_ASSERT_EQUAL(
                invertedSubset,
                NCB::TArraySubsetInvertedIndexing<ui32>(NCB::TFullSubset<ui32>(4))
            );
        }
        {
            TVector<NCB::TIndexRange<ui32>> indexRanges{{6, 9}, {10, 12}, {0, 3}}; //
            NCB::TSavedIndexRanges<ui32> savedIndexRanges(std::move(indexRanges));
            NCB::TArraySubsetIndexing<ui32> subset(NCB::TRangesSubset<ui32>{savedIndexRanges});

            NCB::TArraySubsetInvertedIndexing<ui32> invertedSubset
                = NCB::GetInvertedIndexing(subset, ui32(12), &localExecutor);

            UNIT_ASSERT_EQUAL(
                invertedSubset,
                NCB::TArraySubsetInvertedIndexing<ui32>(
                    NCB::TInvertedIndexedSubset<ui32>(
                        8,
                        TVector<ui32>{
                            5, // 0
                            6, // 1
                            7, // 2
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 3
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 4
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 5
                            0, // 6
                            1, // 7
                            2, // 8
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 9
                            3, // 10
                            4, // 11
                        }
                    )
                )
            );
        }
        {
            NCB::TArraySubsetIndexing<ui32> subset(NCB::TIndexedSubset<ui32>{});
            NCB::TArraySubsetInvertedIndexing<ui32> invertedSubset
                = NCB::GetInvertedIndexing(subset, ui32(0), &localExecutor);
            UNIT_ASSERT_EQUAL(
                invertedSubset,
                NCB::TArraySubsetInvertedIndexing<ui32>(
                    NCB::TInvertedIndexedSubset<ui32>(0, TVector<ui32>{})
                )
            );
        }
        {
            NCB::TArraySubsetIndexing<ui32> subset(NCB::TIndexedSubset<ui32>{3, 1, 5, 4, 7});
            NCB::TArraySubsetInvertedIndexing<ui32> invertedSubset
                = NCB::GetInvertedIndexing(subset, ui32(10), &localExecutor);
            UNIT_ASSERT_EQUAL(
                invertedSubset,
                NCB::TArraySubsetInvertedIndexing<ui32>(
                    NCB::TInvertedIndexedSubset<ui32>(
                        5,
                        TVector<ui32>{
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 0
                            1, // 1
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 2
                            0, // 3
                            3, // 4
                            2, // 5
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 6
                            4, // 7
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT, // 8
                            NCB::TInvertedIndexedSubset<ui32>::NOT_PRESENT // 9
                        }
                    )
                )
            );
        }
    }
}
