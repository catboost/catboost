#include "strip.h"

#include <library/cpp/unittest/registar.h>

#include <util/charset/wide.h>

Y_UNIT_TEST_SUITE(TStripStringTest) {
    Y_UNIT_TEST(TestStrip) {
        struct TTest {
            const char* Str;
            const char* StripLeftRes;
            const char* StripRightRes;
            const char* StripRes;
        };
        static const TTest tests[] = {
            {"  012  ", "012  ", "  012", "012"},
            {"  012", "012", "  012", "012"},
            {"012\t\t", "012\t\t", "012", "012"},
            {"\t012\t", "012\t", "\t012", "012"},
            {"012", "012", "012", "012"},
            {"012\r\n", "012\r\n", "012", "012"},
            {"\n012\r", "012\r", "\n012", "012"},
            {"\n \t\r", "", "", ""},
            {"", "", "", ""},
            {"abc", "abc", "abc", "abc"},
            {"a c", "a c", "a c", "a c"},
        };

        for (const auto& test : tests) {
            TString inputStr(test.Str);

            TString s;
            Strip(inputStr, s);
            UNIT_ASSERT_EQUAL(s, test.StripRes);

            UNIT_ASSERT_EQUAL(StripString(inputStr), test.StripRes);
            UNIT_ASSERT_EQUAL(StripStringLeft(inputStr), test.StripLeftRes);
            UNIT_ASSERT_EQUAL(StripStringRight(inputStr), test.StripRightRes);

            TStringBuf inputStrBuf(test.Str);
            UNIT_ASSERT_EQUAL(StripString(inputStrBuf), test.StripRes);
            UNIT_ASSERT_EQUAL(StripStringLeft(inputStrBuf), test.StripLeftRes);
            UNIT_ASSERT_EQUAL(StripStringRight(inputStrBuf), test.StripRightRes);
        };
    }

    Y_UNIT_TEST(TestCustomStrip) {
        struct TTest {
            const char* Str;
            const char* Result;
        };
        static const TTest tests[] = {
            {"//012//", "012"},
            {"//012", "012"},
            {"012", "012"},
            {"012//", "012"},
        };

        for (auto test : tests) {
            UNIT_ASSERT_EQUAL(
                StripString(TString(test.Str), EqualsStripAdapter('/')),
                test.Result);
        };
    }

    Y_UNIT_TEST(TestCustomStripLeftRight) {
        struct TTest {
            const char* Str;
            const char* ResultLeft;
            const char* ResultRight;
        };
        static const TTest tests[] = {
            {"//012//", "012//", "//012"},
            {"//012", "012", "//012"},
            {"012", "012", "012"},
            {"012//", "012//", "012"},
        };

        for (const auto& test : tests) {
            UNIT_ASSERT_EQUAL(
                StripStringLeft(TString(test.Str), EqualsStripAdapter('/')),
                test.ResultLeft);
            UNIT_ASSERT_EQUAL(
                StripStringRight(TString(test.Str), EqualsStripAdapter('/')),
                test.ResultRight);
        };
    }

    Y_UNIT_TEST(TestNullStringStrip) {
        TStringBuf nullString(nullptr, nullptr);
        UNIT_ASSERT_EQUAL(
            StripString(nullString),
            TString());
    }

    Y_UNIT_TEST(TestWtrokaStrip) {
        UNIT_ASSERT_EQUAL(StripString(AsStringBuf(u" abc ")), u"abc");
        UNIT_ASSERT_EQUAL(StripStringLeft(AsStringBuf(u" abc ")), u"abc ");
        UNIT_ASSERT_EQUAL(StripStringRight(AsStringBuf(u" abc ")), u" abc");
    }

    Y_UNIT_TEST(TestWtrokaCustomStrip) {
        UNIT_ASSERT_EQUAL(
            StripString(
                AsStringBuf(u"/abc/"),
                EqualsStripAdapter(u'/')),
            u"abc");
    }

    Y_UNIT_TEST(TestCollapse) {
        TString s;
        Collapse(TString("  123    456  "), s);
        UNIT_ASSERT(s == " 123 456 ");
        Collapse(TString("  123    456  "), s, 10);
        UNIT_ASSERT(s == " 123 456  ");

        s = TString(" a b c ");
        TString s2 = s;
        Collapse(s2);

        UNIT_ASSERT(s == s2);
        UNIT_ASSERT(s.c_str() == s2.c_str()); // Collapse() does not change the string at all
    }

    Y_UNIT_TEST(TestCollapseText) {
        TString abs1("Very long description string written in unknown language.");
        TString abs2(abs1);
        TString abs3(abs1);
        CollapseText(abs1, 204);
        CollapseText(abs2, 54);
        CollapseText(abs3, 49);
        UNIT_ASSERT_EQUAL(abs1 == "Very long description string written in unknown language.", true);
        UNIT_ASSERT_EQUAL(abs2 == "Very long description string written in unknown ...", true);
        UNIT_ASSERT_EQUAL(abs3 == "Very long description string written in ...", true);
    }
}
