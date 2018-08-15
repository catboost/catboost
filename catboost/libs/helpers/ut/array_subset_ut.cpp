#include <catboost/libs/helpers/array_subset.h>

#include <util/stream/output.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(TArraySubset) {
    void TestOneCase(NCB::TArraySubset<TVector<int>>& arraySubset, const TVector<int>& expectedSubset) {
        size_t expectedIndex = 0;

        UNIT_ASSERT_EQUAL(arraySubset.Size(), expectedSubset.size());

        arraySubset.ForEach([&](size_t index, int value) {
            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
            UNIT_ASSERT_VALUES_EQUAL(expectedSubset[index], value);
            ++expectedIndex;
        });
        UNIT_ASSERT_VALUES_EQUAL(expectedIndex, expectedSubset.size());
    }

    Y_UNIT_TEST(TestFullSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TFullSubset<size_t>{v.size()} );

        NCB::TArraySubset<TVector<int>> arraySubset{v, &arraySubsetIndexing};

        TestOneCase(arraySubset, v);
    }

    Y_UNIT_TEST(TestRangesSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TRangesSubset<size_t>{{{7, 3}, {2,1}, {4, 2}}} );
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Get<NCB::TRangesSubset<size_t>>().Size, 6);

        TVector<int> expectedSubset = {17, 18, 19, 12, 14, 15};
        NCB::TArraySubset<TVector<int>> arraySubset{v, &arraySubsetIndexing};

        TestOneCase(arraySubset, expectedSubset);
    }


    Y_UNIT_TEST(TestIndexedSubset) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TArraySubsetIndexing<size_t> arraySubsetIndexing( NCB::TIndexedSubset<size_t>{6, 5, 2, 0, 1} );
        UNIT_ASSERT_EQUAL(arraySubsetIndexing.Get<NCB::TIndexedSubset<size_t>>().size(), 5);

        TVector<int> expectedSubset = {16, 15, 12, 10, 11};
        NCB::TArraySubset<TVector<int>> arraySubset{v, &arraySubsetIndexing};

        TestOneCase(arraySubset, expectedSubset);
    }
}
