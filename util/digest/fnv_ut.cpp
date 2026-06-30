#include "fnv.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TFnvTest) {
    Y_UNIT_TEST(TestFnv32) {
        const auto h32 = ULL(2849763999);
        UNIT_ASSERT_EQUAL(FnvHash<ui32>("1234567", 7), h32);
        UNIT_ASSERT_EQUAL(FnvHash<ui32>(TStringBuf("1234567")), h32);

        UNIT_ASSERT_EQUAL(FnvHash<ui32>(nullptr, 0), FNV32INIT);
        UNIT_ASSERT_EQUAL(FnvHash<ui32>(TStringBuf()), FNV32INIT);
    }

    Y_UNIT_TEST(TestFnv64) {
        const auto h64 = ULL(2449551094593701855);
        UNIT_ASSERT_EQUAL(FnvHash<ui64>("1234567", 7), h64);
        UNIT_ASSERT_EQUAL(FnvHash<ui64>(TStringBuf("1234567")), h64);

        UNIT_ASSERT_EQUAL(FnvHash<ui64>(nullptr, 0), FNV64INIT);
        UNIT_ASSERT_EQUAL(FnvHash<ui64>(TStringBuf()), FNV64INIT);
    }
} // Y_UNIT_TEST_SUITE(TFnvTest)
