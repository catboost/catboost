#include "crc32c.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TestCrc32c) {
    Y_UNIT_TEST(TestCalc) {
        UNIT_ASSERT_VALUES_EQUAL(Crc32c("abc", 3), ui32(910901175));
    }

    Y_UNIT_TEST(TestUnaligned) {
        const TString str(1000, 'a');
        for (size_t substrLen = 0; substrLen <= str.length(); ++substrLen) {
            const ui32 crc = Crc32c(str.data(), substrLen);
            for (size_t offset = 1; offset + substrLen <= str.length(); ++offset) {
                UNIT_ASSERT_VALUES_EQUAL(Crc32c(str.data() + offset, substrLen), crc);
            }
        }
    }

    Y_UNIT_TEST(TestExtend) {
        UNIT_ASSERT_VALUES_EQUAL(Crc32cExtend(1, "abc", 3), ui32(2466950601));
    }

    Y_UNIT_TEST(TestCombine) {
        ui32 aSum = Crc32c("abc", 3);
        ui32 bSum = Crc32c("de", 2);
        ui32 abSum = Crc32c("abcde", 5);

        UNIT_ASSERT_VALUES_EQUAL(Crc32cCombine(aSum, bSum, 2), abSum);
    }
}
