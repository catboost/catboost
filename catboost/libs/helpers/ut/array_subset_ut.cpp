#include <catboost/libs/helpers/array_subset.h>

#include <util/datetime/base.h>
#include <util/generic/is_in.h>
#include <util/generic/xrange.h>
#include <util/stream/output.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(TArraySubset) {
    Y_UNIT_TEST(TestNullArguments) {
        UNIT_ASSERT_EXCEPTION(
            ([]{
                NCB::TArraySubset<TVector<int>> arraySubset{nullptr, nullptr};
            }()),
            TCatboostException
        );
        UNIT_ASSERT_EXCEPTION(
            ([]{
                TVector<int> v(1, 0);
                NCB::TArraySubset<TVector<int>> arraySubset{&v, nullptr};
            }()),
            TCatboostException
        );
        UNIT_ASSERT_EXCEPTION(
            ([]{
                NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{0} );
                NCB::TArraySubset<TVector<int>> arraySubset{nullptr, &arraySubsetIndexing};
            }()),
            TCatboostException
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
                    localExecutor,
                    [](size_t /*index*/, int /*value*/) { Sleep(TDuration::MilliSeconds(1)); },
                    0
                );
            }()),
            TCatboostException
        );
    }

    void TestOneCase(NCB::TArraySubset<TVector<int>>& arraySubset, const TVector<int>& expectedSubset) {
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
                    localExecutor,
                    [&](size_t index, int value) {
                        UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);

                        auto& indexIterated = indicesIterated.at(index);
                        UNIT_ASSERT(!indexIterated); // each index must be visited only once
                        indexIterated = true;
                    },
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
    }

    Y_UNIT_TEST(TestFullSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{v.size()} );

        NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

        TestOneCase(arraySubset, v);
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
    }


    Y_UNIT_TEST(TestIndexedSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1} );
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Get<NCB::TIndexedSubset<size_t>>().size(), 5);

        TVector<int> expectedSubset = {16, 15, 12, 10, 11};
        NCB::TArraySubset<TVector<int>> arraySubset{&v, &arraySubsetIndexing};

        TestOneCase(arraySubset, expectedSubset);
    }
}
