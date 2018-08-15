#include <catboost/libs/helpers/view_iter.h>

#include <util/stream/output.h>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(TViewIter) {
    void TestOneCase(NCB::TViewIter<TVector<int>, size_t>& viewIter, const TVector<int>& expectedView) {
        size_t expectedIndex = 0;

        UNIT_ASSERT_EQUAL(viewIter.Size(), expectedView.size());

        viewIter.Iter([&](size_t index, int value) {
            UNIT_ASSERT_VALUES_EQUAL(expectedIndex, index);
            UNIT_ASSERT_VALUES_EQUAL(expectedView[index], value);
            ++expectedIndex;
        });
        UNIT_ASSERT_VALUES_EQUAL(expectedIndex, expectedView.size());
    }

    Y_UNIT_TEST(TestFullView) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TViewIndexing<size_t> viewIndexing( NCB::TFullView<size_t>{v.size()} );

        NCB::TViewIter<TVector<int>, size_t> viewIter{v, &viewIndexing};

        TestOneCase(viewIter, v);
    }

    Y_UNIT_TEST(TestBlockView) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TViewIndexing<size_t> viewIndexing( NCB::TBlockView<size_t>{{{7, 3}, {2,1}, {4, 2}}} );
        UNIT_ASSERT_EQUAL(viewIndexing.Get<NCB::TBlockView<size_t>>().Size, 6);

        TVector<int> expectedView = {17, 18, 19, 12, 14, 15};
        NCB::TViewIter<TVector<int>, size_t> viewIter{v, &viewIndexing};

        TestOneCase(viewIter, expectedView);
    }


    Y_UNIT_TEST(TestIndexView) {
        TVector<int> v = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

        NCB::TViewIndexing<size_t> viewIndexing( NCB::TIndexView<size_t>{6, 5, 2, 0, 1} );
        UNIT_ASSERT_EQUAL(viewIndexing.Get<NCB::TIndexView<size_t>>().size(), 5);

        TVector<int> expectedView = {16, 15, 12, 10, 11};
        NCB::TViewIter<TVector<int>, size_t> viewIter{v, &viewIndexing};

        TestOneCase(viewIter, expectedView);
    }
}
