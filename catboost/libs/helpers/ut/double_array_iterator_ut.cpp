#include <catboost/libs/helpers/double_array_iterator.h>

#include <util/generic/algorithm.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>

#include <library/cpp/testing/unittest/registar.h>


using namespace NCB;


Y_UNIT_TEST_SUITE(DoubleArrayIterator) {
    template <class T1, class T2>
    void TestSimple(
        TVector<T1>&& array1,
        TVector<T2>&& array2,
        const TVector<T1>& expectedArray1,
        const TVector<T2>& expectedArray2) {

        TDoubleArrayIterator<T1, T2> beginIter{array1.begin(), array2.begin()};
        TDoubleArrayIterator<T1, T2> endIter{array1.end(), array2.end()};

        Sort(beginIter, endIter, [](auto lhs, auto rhs) { return lhs.first < rhs.first; });

        UNIT_ASSERT_VALUES_EQUAL(array1, expectedArray1);
        UNIT_ASSERT_VALUES_EQUAL(array2, expectedArray2);
    }

    Y_UNIT_TEST(Simple) {
        TestSimple<int, double>(
            {10, 1, 5, 7, 3},
            {10.0, 1.0, 5.0, 7.0, 3.0},
            {1, 3, 5, 7, 10},
            {1.0, 3.0, 5.0, 7.0, 10.0});

        TestSimple<int, TString>(
            {10, 1, 5, 7, 3, 30},
            {"10", "1", "5", "7", "3", "30"},
            {1, 3, 5, 7, 10, 30},
            {"1", "3", "5", "7", "10", "30"});
    }
}
