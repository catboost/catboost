#include "null.h"
#include "printf.h"
#include "str.h"

#include <util/generic/string.h>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TStreamPrintfTest) {
    Y_UNIT_TEST(TestPrintf) {
        TStringStream ss;

        UNIT_ASSERT_EQUAL(Printf(ss, "qw %s %d", "er", 1), 7);
        UNIT_ASSERT_EQUAL(ss.Str(), "qw er 1");
    }

#ifdef __GNUC__
    #pragma GCC diagnostic ignored "-Wformat-zero-length"
#endif // __GNUC__

    Y_UNIT_TEST(TestZeroString) {
        UNIT_ASSERT_EQUAL(Printf(Cnull, ""), 0);
    }

    Y_UNIT_TEST(TestLargePrintf) {
        TString s = NUnitTest::RandomString(1000000);
        TStringStream ss;

        Printf(ss, "%s", s.data());

        UNIT_ASSERT_EQUAL(ss.Str(), s);
    }
} // Y_UNIT_TEST_SUITE(TStreamPrintfTest)
