#include "type.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/wide.h>

Y_UNIT_TEST_SUITE(TStringClassify) {
    Y_UNIT_TEST(TestIsSpace) {
        UNIT_ASSERT_EQUAL(IsSpace(" "), true);
        UNIT_ASSERT_EQUAL(IsSpace("\t\r\n"), true);
        UNIT_ASSERT_EQUAL(IsSpace(""), false);
        UNIT_ASSERT_EQUAL(IsSpace("   a"), false);
    }

    Y_UNIT_TEST(TestIsTrue) {
        UNIT_ASSERT(IsTrue("1"));
        UNIT_ASSERT(IsTrue("yes"));
        UNIT_ASSERT(IsTrue("YeS"));
        UNIT_ASSERT(IsTrue("on"));
        UNIT_ASSERT(IsTrue("true"));
        UNIT_ASSERT(IsTrue("t"));
        UNIT_ASSERT(IsTrue("da"));

        UNIT_ASSERT(!IsTrue(""));
        UNIT_ASSERT(!IsTrue("tr"));
        UNIT_ASSERT(!IsTrue("foobar"));
    }

    Y_UNIT_TEST(TestIsFalse) {
        UNIT_ASSERT(IsFalse("0"));
        UNIT_ASSERT(IsFalse("no"));
        UNIT_ASSERT(IsFalse("off"));
        UNIT_ASSERT(IsFalse("false"));
        UNIT_ASSERT(IsFalse("f"));
        UNIT_ASSERT(IsFalse("net"));

        UNIT_ASSERT(!IsFalse(""));
        UNIT_ASSERT(!IsFalse("fa"));
        UNIT_ASSERT(!IsFalse("foobar"));
    }

    Y_UNIT_TEST(TestIsNumber) {
        UNIT_ASSERT(IsNumber("0"));
        UNIT_ASSERT(IsNumber("12345678901234567890"));
        UNIT_ASSERT(!IsNumber("1234567890a"));
        UNIT_ASSERT(!IsNumber("12345xx67890a"));
        UNIT_ASSERT(!IsNumber("foobar"));
        UNIT_ASSERT(!IsNumber(""));

        UNIT_ASSERT(IsNumber(u"0"));
        UNIT_ASSERT(IsNumber(u"12345678901234567890"));
        UNIT_ASSERT(!IsNumber(u"1234567890a"));
        UNIT_ASSERT(!IsNumber(u"12345xx67890a"));
        UNIT_ASSERT(!IsNumber(u"foobar"));
    }

    Y_UNIT_TEST(TestIsHexNumber) {
        UNIT_ASSERT(IsHexNumber("0"));
        UNIT_ASSERT(IsHexNumber("aaaadddAAAAA"));
        UNIT_ASSERT(IsHexNumber("0123456789ABCDEFabcdef"));
        UNIT_ASSERT(IsHexNumber("12345678901234567890"));
        UNIT_ASSERT(IsHexNumber("1234567890a"));
        UNIT_ASSERT(!IsHexNumber("12345xx67890a"));
        UNIT_ASSERT(!IsHexNumber("foobar"));
        UNIT_ASSERT(!IsHexNumber(TString()));

        UNIT_ASSERT(IsHexNumber(u"0"));
        UNIT_ASSERT(IsHexNumber(u"aaaadddAAAAA"));
        UNIT_ASSERT(IsHexNumber(u"0123456789ABCDEFabcdef"));
        UNIT_ASSERT(IsHexNumber(u"12345678901234567890"));
        UNIT_ASSERT(IsHexNumber(u"1234567890a"));
        UNIT_ASSERT(!IsHexNumber(u"12345xx67890a"));
        UNIT_ASSERT(!IsHexNumber(u"foobar"));
        UNIT_ASSERT(!IsHexNumber(TUtf16String()));
    }
} // Y_UNIT_TEST_SUITE(TStringClassify)
