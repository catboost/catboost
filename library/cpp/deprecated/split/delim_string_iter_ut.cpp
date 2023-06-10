#include "delim_string_iter.h"
#include <util/generic/vector.h>
#include <library/cpp/testing/unittest/registar.h>

/// Test that TDelimStringIter build on top of given string and delimeter will produce expected sequence
static void AssertStringSplit(const TString& str, const TString& delim, const TVector<TString>& expected) {
    TDelimStringIter it(str, delim);

    // test iterator invariants
    for (const auto& expectedString : expected) {
        UNIT_ASSERT(it.Valid());
        UNIT_ASSERT(bool(it));
        UNIT_ASSERT_STRINGS_EQUAL(*it, expectedString);
        ++it;
    }
    UNIT_ASSERT(!it.Valid());
}

Y_UNIT_TEST_SUITE(TDelimStrokaIterTestSuite) {
    Y_UNIT_TEST(SingleCharacterAsDelimiter) {
        AssertStringSplit(
            "Hello words!", " ", {"Hello", "words!"});
    }

    Y_UNIT_TEST(MultipleCharactersAsDelimiter) {
        AssertStringSplit(
            "0, 1, 1, 2, 3, 5, 8, 13, 21, 34", "1, ", {"0, ", "", "2, 3, 5, 8, 13, 2", "34"});
    }

    Y_UNIT_TEST(NoDelimitersPresent) {
        AssertStringSplit("This string could be yours", "\t", {"This string could be yours"});
    }

    Y_UNIT_TEST(Cdr) {
        TDelimStringIter it("a\tc\t", "\t");
        UNIT_ASSERT_STRINGS_EQUAL(*it, "a");
        UNIT_ASSERT_STRINGS_EQUAL(it.Cdr(), "c\t");
        ++it;
        UNIT_ASSERT_STRINGS_EQUAL(it.Cdr(), "");
    }

    Y_UNIT_TEST(ForIter) {
        TVector<TStringBuf> expected = {"1", "", "3@4", ""};
        TVector<TStringBuf> got;

        for (TStringBuf x : TDelimStroka("1@@@@3@4@@", "@@")) {
            got.push_back(x);
        }

        UNIT_ASSERT_EQUAL(got, expected);
    }
}

static void AssertKeyValueStringSplit(
    const TStringBuf str,
    const TStringBuf delim,
    const TVector<std::pair<TStringBuf, TStringBuf>>& expected) {
    TKeyValueDelimStringIter it(str, delim);

    for (const auto& expectedKeyValue : expected) {
        UNIT_ASSERT(it.Valid());
        UNIT_ASSERT_STRINGS_EQUAL(it.Key(), expectedKeyValue.first);
        UNIT_ASSERT_STRINGS_EQUAL(it.Value(), expectedKeyValue.second);
        ++it;
    }
    UNIT_ASSERT(!it.Valid());
}

Y_UNIT_TEST_SUITE(TKeyValueDelimStringIterTestSuite) {
    Y_UNIT_TEST(SingleCharacterAsDelimiter) {
        AssertKeyValueStringSplit(
            "abc=123,cde=qwer", ",",
            {{"abc", "123"},
             {"cde", "qwer"}});
    }

    Y_UNIT_TEST(MultipleCharactersAsDelimiter) {
        AssertKeyValueStringSplit(
            "abc=xyz@@qwerty=zxcv", "@@",
            {{"abc", "xyz"},
             {"qwerty", "zxcv"}});
    }

    Y_UNIT_TEST(NoDelimiters) {
        AssertKeyValueStringSplit(
            "abc=zz", ",",
            {{"abc", "zz"}});
    }

    Y_UNIT_TEST(EmptyElements) {
        AssertKeyValueStringSplit(
            "@@abc=zxy@@@@qwerty=y@@", "@@",
            {{"", ""},
             {"abc", "zxy"},
             {"", ""},
             {"qwerty", "y"},
             {"", ""}});
    }
}
