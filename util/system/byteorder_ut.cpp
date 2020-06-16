#include "byteorder.h"

#include <library/cpp/testing/unittest/registar.h>

class TByteOrderTest: public TTestBase {
    UNIT_TEST_SUITE(TByteOrderTest);
    UNIT_TEST(TestSwap16)
    UNIT_TEST(TestSwap32)
    UNIT_TEST(TestSwap64)
    UNIT_TEST_SUITE_END();

private:
    inline void TestSwap16() {
        UNIT_ASSERT_EQUAL((ui16)0x1234, SwapBytes((ui16)0x3412));
    }

    inline void TestSwap32() {
        UNIT_ASSERT_EQUAL(0x12345678, SwapBytes(0x78563412));
    }

    inline void TestSwap64() {
        UNIT_ASSERT_EQUAL(0x1234567890abcdefULL, SwapBytes((ui64)ULL(0xefcdab9078563412)));
    }
};

UNIT_TEST_SUITE_REGISTRATION(TByteOrderTest);
