#include "printf.h"

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TStringPrintf) {
    SIMPLE_UNIT_TEST(TestSprintf) {
        TString s;
        int len = sprintf(s, "Hello %s", "world");
        UNIT_ASSERT_EQUAL(s, TString("Hello world"));
        UNIT_ASSERT_EQUAL(len, 11);
    }

    SIMPLE_UNIT_TEST(TestFcat) {
        TString s;
        int len = sprintf(s, "Hello %s", "world");
        UNIT_ASSERT_EQUAL(s, TString("Hello world"));
        UNIT_ASSERT_EQUAL(len, 11);
        len = fcat(s, " qwqw%s", "as");
        UNIT_ASSERT_EQUAL(s, TString("Hello world qwqwas"));
        UNIT_ASSERT_EQUAL(len, 7);
    }

    SIMPLE_UNIT_TEST(TestSpecial) {
        UNIT_ASSERT("4294967295" == Sprintf("%" PRIu32, (ui32)(-1)));
    }
}
