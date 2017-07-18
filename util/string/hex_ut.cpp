#include "hex.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(THexCodingTest) {
    SIMPLE_UNIT_TEST(TestEncode) {
        UNIT_ASSERT_EQUAL(HexEncode("i1634iqwbf,&msdb"), "693136333469717762662C266D736462");
    }

    SIMPLE_UNIT_TEST(TestDecode) {
        UNIT_ASSERT_EQUAL(HexDecode("693136333469717762662C266D736462"), "i1634iqwbf,&msdb");
    }

    SIMPLE_UNIT_TEST(TestDecodeCase) {
        UNIT_ASSERT_EQUAL(HexDecode("12ABCDEF"), HexDecode("12abcdef"));
        UNIT_ASSERT_EXCEPTION(HexDecode("Hello"), yexception); //< incorrect chars
        UNIT_ASSERT_EXCEPTION(HexDecode("123"), yexception);   //< odd length
    }
}
