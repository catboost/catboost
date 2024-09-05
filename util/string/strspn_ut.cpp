#include "strspn.h"

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TStrSpnTest) {
    Y_UNIT_TEST(FindFirstOf) {
        const TString s("some text!");

        UNIT_ASSERT_EQUAL(TCompactStrSpn("mos").FindFirstOf(s.begin(), s.end()), s.begin());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("mos").FindFirstOf(s.c_str()), s.begin());

        UNIT_ASSERT_EQUAL(TCompactStrSpn("xt").FindFirstOf(s.begin(), s.end()), s.begin() + 5);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("xt").FindFirstOf(s.c_str()), s.begin() + 5);

        UNIT_ASSERT_EQUAL(TCompactStrSpn(".?!").FindFirstOf(s.begin(), s.end()), s.end() - 1);
        UNIT_ASSERT_EQUAL(TCompactStrSpn(".?!").FindFirstOf(s.c_str()), s.end() - 1);

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(s.begin(), s.end()), s.end());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(s.c_str()), s.end());

        // Must be const. If not, non-const begin() will clone the shared empty string
        // and the next assertion will possibly use invalidated end iterator.
        const TString empty;

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(empty.begin(), empty.end()), empty.end());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(empty.c_str()), empty.end());
    }

    Y_UNIT_TEST(FindFirstNotOf) {
        const TString s("abacabaxyz");

        UNIT_ASSERT_EQUAL(TCompactStrSpn("123").FindFirstNotOf(s.begin(), s.end()), s.begin());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("123").FindFirstNotOf(s.c_str()), s.begin());

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(s.begin(), s.end()), s.begin() + 7);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(s.c_str()), s.begin() + 7);

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxy").FindFirstNotOf(s.begin(), s.end()), s.end() - 1);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxy").FindFirstNotOf(s.c_str()), s.end() - 1);

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxyz").FindFirstNotOf(s.begin(), s.end()), s.end());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxyz").FindFirstNotOf(s.c_str()), s.end());

        const TString empty;

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(empty.begin(), empty.end()), empty.end());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(empty.c_str()), empty.end());
    }

    Y_UNIT_TEST(FindFirstOfReverse) {
        TStringBuf s("some text");

        UNIT_ASSERT_EQUAL(TCompactStrSpn("xt").FindFirstOf(s.rbegin(), s.rend()), s.rbegin());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("mos").FindFirstOf(s.rbegin(), s.rend()), s.rend() - 3);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("s").FindFirstOf(s.rbegin(), s.rend()), s.rend() - 1);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(s.rbegin(), s.rend()), s.rend());

        TStringBuf empty;
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstOf(empty.rbegin(), empty.rend()), empty.rend());
    }

    Y_UNIT_TEST(FindFirstNotOfReverse) {
        TStringBuf s("_abacabaxyz");

        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(s.rbegin(), s.rend()), s.rbegin());
        UNIT_ASSERT_EQUAL(TCompactStrSpn("xyz").FindFirstNotOf(s.rbegin(), s.rend()), s.rbegin() + 3);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxyz").FindFirstNotOf(s.rbegin(), s.rend()), s.rend() - 1);
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abcxyz_").FindFirstNotOf(s.rbegin(), s.rend()), s.rend());

        TStringBuf empty;
        UNIT_ASSERT_EQUAL(TCompactStrSpn("abc").FindFirstNotOf(empty.rbegin(), empty.rend()), empty.rend());
    }
} // Y_UNIT_TEST_SUITE(TStrSpnTest)
