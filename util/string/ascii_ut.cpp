#include "ascii.h"
#include <ctype.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TAsciiTest) {
    Y_UNIT_TEST(TestAscii) {
        UNIT_ASSERT(IsAsciiDigit('3'));
        UNIT_ASSERT(!IsAsciiDigit('x'));

        UNIT_ASSERT(IsAsciiAlpha('r'));
        UNIT_ASSERT(IsAsciiAlpha('R'));
        UNIT_ASSERT(!IsAsciiAlpha('3'));

        UNIT_ASSERT_EQUAL(AsciiToLower('3'), '3');
        UNIT_ASSERT_EQUAL(AsciiToLower('A'), 'a');
        UNIT_ASSERT_EQUAL(AsciiToLower('a'), 'a');

        UNIT_ASSERT_EQUAL(AsciiToUpper('3'), '3');
        UNIT_ASSERT_EQUAL(AsciiToUpper('A'), 'A');
        UNIT_ASSERT_EQUAL(AsciiToUpper('a'), 'A');

        UNIT_ASSERT(IsAscii('a'));
        UNIT_ASSERT(!IsAscii(-100));
        UNIT_ASSERT(!IsAscii(+200));
        UNIT_ASSERT(!IsAscii(int('a') + 256));

        for (int i = 0; i < 128; ++i) {
            UNIT_ASSERT_VALUES_EQUAL((bool)isxdigit(i), IsAsciiHex(i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isspace(i), IsAsciiSpace((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isspace(i), IsAsciiSpace((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isalnum(i), IsAsciiAlnum((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isalpha(i), IsAsciiAlpha((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isupper(i), IsAsciiUpper((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)islower(i), IsAsciiLower((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)isdigit(i), IsAsciiDigit((char)i));
            UNIT_ASSERT_VALUES_EQUAL((bool)ispunct(i), IsAsciiPunct((char)i));
        }
    }

    Y_UNIT_TEST(Test1) {
        for (int i = 128; i < 1000; ++i) {
            UNIT_ASSERT(!IsAsciiHex(i));
            UNIT_ASSERT(!IsAsciiSpace(i));
            UNIT_ASSERT(!IsAsciiAlnum(i));
            UNIT_ASSERT(!IsAsciiAlpha(i));
            UNIT_ASSERT(!IsAsciiUpper(i));
            UNIT_ASSERT(!IsAsciiLower(i));
            UNIT_ASSERT(!IsAsciiDigit(i));
            UNIT_ASSERT(!IsAsciiPunct(i));
        }

        for (int i = -1000; i < 0; ++i) {
            UNIT_ASSERT(!IsAsciiHex(i));
            UNIT_ASSERT(!IsAsciiSpace(i));
            UNIT_ASSERT(!IsAsciiAlnum(i));
            UNIT_ASSERT(!IsAsciiAlpha(i));
            UNIT_ASSERT(!IsAsciiUpper(i));
            UNIT_ASSERT(!IsAsciiLower(i));
            UNIT_ASSERT(!IsAsciiDigit(i));
            UNIT_ASSERT(!IsAsciiPunct(i));
        }
    }

    Y_UNIT_TEST(CompareTest) {
        UNIT_ASSERT(AsciiEqualsIgnoreCase("qqq", "qQq"));
        UNIT_ASSERT(AsciiEqualsIgnoreCase("qqq", TStringBuf("qQq")));
        TString qq = "qq";
        TString qQ = "qQ";
        UNIT_ASSERT(AsciiEqualsIgnoreCase(qq, qQ));

        TString x = "qqqA";
        TString y = "qQqB";
        TString z = "qQnB";
        TString zz = "qQqq";
        TString zzz = "qQqqq";
        TStringBuf xs = TStringBuf(x.data(), 3);
        TStringBuf ys = TStringBuf(y.data(), 3);
        TStringBuf zs = TStringBuf(z.data(), 3);
        UNIT_ASSERT(AsciiCompareIgnoreCase(xs, ys) == 0);
        UNIT_ASSERT(AsciiCompareIgnoreCase(xs, zs) > 0);
        UNIT_ASSERT(AsciiCompareIgnoreCase(xs, zz) < 0);
        UNIT_ASSERT(AsciiCompareIgnoreCase(zzz, zz) > 0);

        UNIT_ASSERT(AsciiCompareIgnoreCase("qqQ", "qq") > 0);
        UNIT_ASSERT(AsciiCompareIgnoreCase("qq", "qq") == 0);

        UNIT_ASSERT_EQUAL(AsciiHasPrefix("qweasd", "qwe"), true);
        UNIT_ASSERT_EQUAL(AsciiHasPrefix("qweasd", "qWe"), false);
        UNIT_ASSERT_EQUAL(AsciiHasPrefix("qweasd", "eWq"), false);

        UNIT_ASSERT_EQUAL(AsciiHasPrefixIgnoreCase("qweasd", "qWe"), true);
        UNIT_ASSERT_EQUAL(AsciiHasPrefixIgnoreCase("qweasd", "eWq"), false);

        UNIT_ASSERT_EQUAL(AsciiHasSuffixIgnoreCase("qweasd", "asD"), true);
        UNIT_ASSERT_EQUAL(AsciiHasSuffixIgnoreCase("qweasd", "ast"), false);
    }
} // Y_UNIT_TEST_SUITE(TAsciiTest)
