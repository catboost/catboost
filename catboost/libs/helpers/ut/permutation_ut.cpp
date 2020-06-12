#include <catboost/libs/helpers/permutation.h>

#include <util/system/types.h>

#include <library/cpp/testing/unittest/registar.h>


Y_UNIT_TEST_SUITE(Permutation) {
    Y_UNIT_TEST(IsPermutation) {
        UNIT_ASSERT(IsPermutation(TVector<int>()));
        UNIT_ASSERT(IsPermutation(TVector<int>{0}));
        UNIT_ASSERT(IsPermutation(TVector<int>{0, 1}));
        UNIT_ASSERT(!IsPermutation(TVector<int>{1}));
        UNIT_ASSERT(!IsPermutation(TVector<ui8>{0, 0}));
        UNIT_ASSERT(IsPermutation(TVector<ui16>{0, 1, 2}));
        UNIT_ASSERT(IsPermutation(TVector<ui32>{2, 1, 0}));
        UNIT_ASSERT(IsPermutation(TVector<i32>{1, 0, 2}));
        UNIT_ASSERT(IsPermutation(TVector<i32>{1, 0, 2, 3}));
        UNIT_ASSERT(IsPermutation(TVector<ui64>{1, 2, 0, 4, 3, 5}));
        UNIT_ASSERT(!IsPermutation(TVector<ui64>{1, 2, 0, 0, 3, 5}));
        UNIT_ASSERT(!IsPermutation(TVector<i64>{0, 2, 3}));
        UNIT_ASSERT(!IsPermutation(TVector<i64>{1, 2, 4, 3, 5}));
    }
}
