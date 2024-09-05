#include "format.h"

#include <library/cpp/testing/unittest/registar.h>

#include <sstream>

Y_UNIT_TEST_SUITE(StdOstreamFormattingTest) {
    template <typename T>
    TString ToStringViaOstream(T baseNumber) {
        std::stringstream ss;
        ss << baseNumber;
        return ss.str();
    }

    Y_UNIT_TEST(TestBin) {
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui32>(2), nullptr)), "10");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(SBin(static_cast<i32>(-2), nullptr)), "-10");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(SBin(static_cast<i32>(-2))), "-0b00000000000000000000000000000010");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(SBin(static_cast<i32>(-2), HF_FULL)), "-00000000000000000000000000000010");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui32>(15), nullptr)), "1111");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui32>(1))), "0b00000000000000000000000000000001");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui32>(-1))), "0b11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<i32>(-1))), "0b11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<i32>(-1), nullptr)), "11111111111111111111111111111111");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui32>(256))), "0b00000000000000000000000100000000");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui8>(16))), "0b00010000");
        UNIT_ASSERT_VALUES_EQUAL(ToStringViaOstream(Bin(static_cast<ui64>(1234587912357ull))), "0b0000000000000000000000010001111101110011001011001000100010100101");
    }
} // Y_UNIT_TEST_SUITE(StdOstreamFormattingTest)
