#include "strip.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/charset/wide.h>

Y_UNIT_TEST_SUITE(TStripStringTest) {
    struct TStripTest {
        TStringBuf Str;
        TStringBuf StripLeftRes;
        TStringBuf StripRightRes;
        TStringBuf StripRes;
    };
    static constexpr TStripTest StripTests[] = {
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
        {"  abc  ", "abc  ", "  abc", "abc"},
        {"a c", "a c", "a c", "a c"},
        {"  long string to avoid SSO            \n", "long string to avoid SSO            \n", "  long string to avoid SSO", "long string to avoid SSO"},
        {"  набор не-ascii букв  ", "набор не-ascii букв  ", "  набор не-ascii букв", "набор не-ascii букв"},
        // Russian "х" ends with \x85, whis is a space character in some encodings.
        {"последней буквой идет х ", "последней буквой идет х ", "последней буквой идет х", "последней буквой идет х"},
    };

    Y_UNIT_TEST(TestStrip) {
        for (const auto& test : StripTests) {
            TString inputStr(test.Str);

            TString s;
            Strip(inputStr, s);
            UNIT_ASSERT_VALUES_EQUAL(s, test.StripRes);

            UNIT_ASSERT_VALUES_EQUAL(StripString(inputStr), test.StripRes);
            UNIT_ASSERT_VALUES_EQUAL(StripStringLeft(inputStr), test.StripLeftRes);
            UNIT_ASSERT_VALUES_EQUAL(StripStringRight(inputStr), test.StripRightRes);

            TStringBuf inputStrBuf(test.Str);
            UNIT_ASSERT_VALUES_EQUAL(StripString(inputStrBuf), test.StripRes);
            UNIT_ASSERT_VALUES_EQUAL(StripStringLeft(inputStrBuf), test.StripLeftRes);
            UNIT_ASSERT_VALUES_EQUAL(StripStringRight(inputStrBuf), test.StripRightRes);
        };
    }

    Y_UNIT_TEST(TestStripInPlace) {
        // On Darwin default locale is set to a value which interprets certain cyrillic utf-8 sequences as spaces.
        // Which we do not use ::isspace and only strip ASCII spaces, we want to ensure that this will not change in the future.
        std::setlocale(LC_ALL, "");
        for (const auto& test : StripTests) {
            TString str(test.Str);
            Y_ASSERT(str.IsDetached() || str.empty()); // prerequisite of the test; check that we don't try to modify shared COW-string in-place by accident
            const void* stringPtrPrior = str.data();
            StripInPlace(str);
            const void* stringPtrAfter = str.data();
            UNIT_ASSERT_VALUES_EQUAL(str, test.StripRes);
            if (!test.Str.empty()) {
                UNIT_ASSERT_EQUAL_C(stringPtrPrior, stringPtrAfter, TString(test.Str).Quote()); // StripInPlace should reuse buffer of original string
            }
        }
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
                StripStringLeft(TStringBuf(test.Str), EqualsStripAdapter('/')),
                test.ResultLeft);
            UNIT_ASSERT_EQUAL(
                StripStringLeft(std::string_view(test.Str), EqualsStripAdapter('/')),
                test.ResultLeft);
            UNIT_ASSERT_EQUAL(
                StripStringRight(TString(test.Str), EqualsStripAdapter('/')),
                test.ResultRight);
            UNIT_ASSERT_EQUAL(
                StripStringRight(TStringBuf(test.Str), EqualsStripAdapter('/')),
                test.ResultRight);
            UNIT_ASSERT_EQUAL(
                StripStringRight(std::string_view(test.Str), EqualsStripAdapter('/')),
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
        UNIT_ASSERT_EQUAL(StripString(TWtringBuf(u" abc ")), u"abc");
        UNIT_ASSERT_EQUAL(StripStringLeft(TWtringBuf(u" abc ")), u"abc ");
        UNIT_ASSERT_EQUAL(StripStringRight(TWtringBuf(u" abc ")), u" abc");
    }

    Y_UNIT_TEST(TestWtrokaCustomStrip) {
        UNIT_ASSERT_EQUAL(
            StripString(
                TWtringBuf(u"/abc/"),
                EqualsStripAdapter(u'/')),
            u"abc");
    }

    Y_UNIT_TEST(TestSelfRefStringStrip) {
        TStringBuf sb = "  abc ";
        StripString(sb, sb);
        UNIT_ASSERT_EQUAL(sb, "abc");

        TString str = "  abc ";
        StripString(str, str);
        UNIT_ASSERT_EQUAL(str, "abc");
    }

    Y_UNIT_TEST(TestCollapseUtf32) {
        TUtf32String s;
        Collapse(UTF8ToUTF32<true>("  123    456  "), s, IsWhitespace);
        UNIT_ASSERT(s == UTF8ToUTF32<true>(" 123 456 "));
        Collapse(UTF8ToUTF32<true>("  123    456  "), s, IsWhitespace, 10);
        UNIT_ASSERT(s == UTF8ToUTF32<true>(" 123 456  "));

        s = UTF8ToUTF32<true>(" a b c ");
        TUtf32String s2 = s;
        CollapseInPlace(s2, IsWhitespace);

        UNIT_ASSERT(s == s2);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == s2.c_str()); // Collapse() does not change the string at all
#endif
    }

    Y_UNIT_TEST(TestCollapseUtf16) {
        TUtf16String s;
        Collapse(UTF8ToWide<true>("  123    456  "), s);
        UNIT_ASSERT(s == UTF8ToWide<true>(" 123 456 "));
        Collapse(UTF8ToWide<true>("  123    456  "), s, 10);
        UNIT_ASSERT(s == UTF8ToWide<true>(" 123 456  "));

        s = UTF8ToWide<true>(" a b c ");
        TUtf16String s2 = s;
        CollapseInPlace(s2);

        UNIT_ASSERT(s == s2);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == s2.c_str()); // Collapse() does not change the string at all
#endif
    }

    Y_UNIT_TEST(TestCollapse) {
        TString s;
        Collapse(TString("  123    456  "), s);
        UNIT_ASSERT(s == " 123 456 ");
        Collapse(TString("  123    456  "), s, 10);
        UNIT_ASSERT(s == " 123 456  ");

        s = TString(" a b c ");
        TString s2 = s;
        CollapseInPlace(s2);

        UNIT_ASSERT(s == s2);
#ifndef TSTRING_IS_STD_STRING
        UNIT_ASSERT(s.c_str() == s2.c_str()); // Collapse() does not change the string at all
#endif
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
} // Y_UNIT_TEST_SUITE(TStripStringTest)
