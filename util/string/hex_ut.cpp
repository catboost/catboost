#include "hex.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(THexCodingTest) {
    Y_UNIT_TEST(TestEncode) {
        UNIT_ASSERT_EQUAL(HexEncode("i1634iqwbf,&msdb"), "693136333469717762662C266D736462");
    }

    Y_UNIT_TEST(TestDecode) {
        UNIT_ASSERT_EQUAL(HexDecode("693136333469717762662C266D736462"), "i1634iqwbf,&msdb");
    }

    Y_UNIT_TEST(TestDecodeCase) {
        UNIT_ASSERT_EQUAL(HexDecode("12ABCDEF"), HexDecode("12abcdef"));
        UNIT_ASSERT_EXCEPTION(HexDecode("Hello"), yexception); //< incorrect chars
        UNIT_ASSERT_EXCEPTION(HexDecode("123"), yexception);   //< odd length
    }
} // Y_UNIT_TEST_SUITE(THexCodingTest)
