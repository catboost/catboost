#include "type.h"

#include <library/unittest/registar.h>

#include <util/charset/wide.h>

SIMPLE_UNIT_TEST_SUITE(TStringClassify) {
    SIMPLE_UNIT_TEST(TestIsSpace) {
        UNIT_ASSERT_EQUAL(IsSpace(" "), true);
        UNIT_ASSERT_EQUAL(IsSpace("\t\r\n"), true);
        UNIT_ASSERT_EQUAL(IsSpace(""), false);
        UNIT_ASSERT_EQUAL(IsSpace("   a"), false);
    }

    SIMPLE_UNIT_TEST(TestIsTrue) {
        UNIT_ASSERT(IsTrue("1"));
        UNIT_ASSERT(IsTrue("yes"));
        UNIT_ASSERT(IsTrue("YeS"));
        UNIT_ASSERT(IsTrue("on"));
        UNIT_ASSERT(IsTrue("true"));
        UNIT_ASSERT(IsTrue("da"));

        UNIT_ASSERT(!IsTrue("")); // IsTrue won't return true on empty strings anymore

        UNIT_ASSERT(!IsTrue("foobar"));
    }

    SIMPLE_UNIT_TEST(TestIsFalse) {
        UNIT_ASSERT(IsFalse("0"));
        UNIT_ASSERT(IsFalse("no"));
        UNIT_ASSERT(IsFalse("off"));
        UNIT_ASSERT(IsFalse("false"));
        UNIT_ASSERT(IsFalse("net"));

        UNIT_ASSERT(!IsFalse("")); // IsFalse won't return true on empty strings anymore

        UNIT_ASSERT(!IsFalse("foobar"));
    }

    SIMPLE_UNIT_TEST(TestIsNumber) {
        UNIT_ASSERT(IsNumber("0"));
        UNIT_ASSERT(IsNumber("12345678901234567890"));
        UNIT_ASSERT(!IsNumber("1234567890a"));
        UNIT_ASSERT(!IsNumber("12345xx67890a"));
        UNIT_ASSERT(!IsNumber("foobar"));

        UNIT_ASSERT(IsNumber(UTF8ToWide("0")));
        UNIT_ASSERT(IsNumber(UTF8ToWide("12345678901234567890")));
        UNIT_ASSERT(!IsNumber(UTF8ToWide("1234567890a")));
        UNIT_ASSERT(!IsNumber(UTF8ToWide("12345xx67890a")));
        UNIT_ASSERT(!IsNumber(UTF8ToWide("foobar")));
    }

    SIMPLE_UNIT_TEST(TestIsHexNumber) {
        UNIT_ASSERT(IsHexNumber("0"));
        UNIT_ASSERT(IsHexNumber("aaaadddAAAAA"));
        UNIT_ASSERT(IsHexNumber("0123456789ABCDEFabcdef"));
        UNIT_ASSERT(IsHexNumber("12345678901234567890"));
        UNIT_ASSERT(IsHexNumber("1234567890a"));
        UNIT_ASSERT(!IsHexNumber("12345xx67890a"));
        UNIT_ASSERT(!IsHexNumber("foobar"));
        UNIT_ASSERT(!IsHexNumber(TString()));

        UNIT_ASSERT(IsHexNumber(UTF8ToWide("0")));
        UNIT_ASSERT(IsHexNumber(UTF8ToWide("aaaadddAAAAA")));
        UNIT_ASSERT(IsHexNumber(UTF8ToWide("0123456789ABCDEFabcdef")));
        UNIT_ASSERT(IsHexNumber(UTF8ToWide("12345678901234567890")));
        UNIT_ASSERT(IsHexNumber(UTF8ToWide("1234567890a")));
        UNIT_ASSERT(!IsHexNumber(UTF8ToWide("12345xx67890a")));
        UNIT_ASSERT(!IsHexNumber(UTF8ToWide("foobar")));
        UNIT_ASSERT(!IsHexNumber(TUtf16String()));
    }
}
