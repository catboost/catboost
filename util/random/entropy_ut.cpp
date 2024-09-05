#include "entropy.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestEntropy) {
    Y_UNIT_TEST(TestSeed) {
        char buf[100];

        for (size_t i = 0; i < sizeof(buf); ++i) {
            Seed().LoadOrFail(buf, i);
        }
    }
} // Y_UNIT_TEST_SUITE(TestEntropy)
