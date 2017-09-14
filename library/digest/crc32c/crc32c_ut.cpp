#include "crc32c.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TestCrc32c) {
    SIMPLE_UNIT_TEST(TestCalc) {
        UNIT_ASSERT_VALUES_EQUAL(Crc32c("abc", 3), ui32(910901175));
    }

    SIMPLE_UNIT_TEST(TestExtend) {
        UNIT_ASSERT_VALUES_EQUAL(Crc32cExtend(1, "abc", 3), ui32(2466950601));
    }
}
