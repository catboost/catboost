#include "sequence.h"

#include <library/unittest/registar.h>
#include <util/generic/vector.h>

class TRangeHashTest: public TTestBase {
    UNIT_TEST_SUITE(TRangeHashTest);
    UNIT_TEST(TestStrokaInt)
    UNIT_TEST(TestIntVector)
    UNIT_TEST(TestOneElement)
    UNIT_TEST(TestCollectionIndependancy);
    UNIT_TEST_SUITE_END();

private:
    inline void TestStrokaInt() {
        const size_t canonicalHash = ULL(12727184940294366172);
        UNIT_ASSERT_EQUAL(canonicalHash, TRangeHash<>()(TString("12345")));
    }

    inline void TestIntVector() {
        const size_t canonicalHash = ULL(1351128487744230578);
        yvector<int> testVec = {1, 2, 4, 3};
        UNIT_ASSERT_EQUAL(canonicalHash, TRangeHash<>()(testVec));
    }

    inline void TestOneElement() {
        const int testVal = 42;
        yvector<int> testVec = {testVal};
        UNIT_ASSERT_UNEQUAL(THash<int>()(testVal), TRangeHash<>()(testVec));
    }

    inline void TestCollectionIndependancy() {
        yvector<char> testVec = {'a', 'b', 'c'};
        TString testStroka = "abc";
        UNIT_ASSERT_EQUAL(TRangeHash<>()(testVec), TRangeHash<>()(testStroka));
    }
};

class TSequenceHashTest: public TTestBase {
    UNIT_TEST_SUITE(TSequenceHashTest);
    UNIT_TEST(TestSimpleBuffer)
    UNIT_TEST_SUITE_END();

private:
    inline void TestSimpleBuffer() {
        int arr[] = {1, 2, 3};
        const size_t canonicalHash = ULL(3903918011533391876);
        TContiguousHash<TSimpleRangeHash> hasher;
        UNIT_ASSERT_EQUAL(canonicalHash, hasher(NArrayRef::TArrayRef<int>(arr, arr + 3)));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TRangeHashTest);
UNIT_TEST_SUITE_REGISTRATION(TSequenceHashTest);
