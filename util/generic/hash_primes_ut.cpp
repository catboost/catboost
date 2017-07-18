#include "hash_primes.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestHashPrimes) {
    SIMPLE_UNIT_TEST(Test1) {
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(1), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(6), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(7), 7);
        UNIT_ASSERT_VALUES_EQUAL(HashBucketCount(8), 17);
    }
}
