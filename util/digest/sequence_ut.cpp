#include "sequence.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>

class TRangeHashTest: public TTestBase {
    UNIT_TEST_SUITE(TRangeHashTest);
    UNIT_TEST(TestStrokaInt)
    UNIT_TEST(TestIntVector)
    UNIT_TEST(TestOneElement)
    UNIT_TEST(TestMap);
    UNIT_TEST(TestCollectionIndependancy);
    UNIT_TEST_SUITE_END();

private:
    inline void TestStrokaInt() {
        const size_t canonicalHash = static_cast<size_t>(ULL(12727184940294366172));
        UNIT_ASSERT_EQUAL(canonicalHash, TRangeHash<>()(TString("12345")));
    }

    inline void TestIntVector() {
        const size_t canonicalHash = static_cast<size_t>(ULL(1351128487744230578));
        TVector<int> testVec = {1, 2, 4, 3};
        UNIT_ASSERT_EQUAL(canonicalHash, TRangeHash<>()(testVec));
    }

    inline void TestOneElement() {
        const int testVal = 42;
        TVector<int> testVec = {testVal};
        UNIT_ASSERT_UNEQUAL(THash<int>()(testVal), TRangeHash<>()(testVec));
    }

    inline void TestMap() {
        const size_t canonicalHash = static_cast<size_t>(ULL(4415387926488545605));
        TMap<TString, int> testMap{{"foo", 123}, {"bar", 456}};
        UNIT_ASSERT_EQUAL(canonicalHash, TRangeHash<>()(testMap));
    }

    inline void TestCollectionIndependancy() {
        TVector<char> testVec = {'a', 'b', 'c'};
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
        const size_t canonicalHash = static_cast<size_t>(ULL(3903918011533391876));
        TContiguousHash<TSimpleRangeHash> hasher;
        UNIT_ASSERT_EQUAL(canonicalHash, hasher(TArrayRef<int>(arr, arr + 3)));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TRangeHashTest);
UNIT_TEST_SUITE_REGISTRATION(TSequenceHashTest);
