#include "entropy.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestEntropy) {
    SIMPLE_UNIT_TEST(TestSeed) {
        char buf[100];

        for (size_t i = 0; i < sizeof(buf); ++i) {
            Seed().LoadOrFail(buf, i);
        }
    }
}
