#include "delim_stroka_iter.h"
#include <util/generic/vector.h>
#include <library/unittest/registar.h>

/// Test that TDelimStrokaIter build on top of given string and delimeter will produce expected sequence
static void AssertStringSplit(const TString& str, const TString& delim, const yvector<TString>& expected) {
    TDelimStrokaIter it(str, delim);

    // test iterator invariants
    for (const auto& expectedString : expected) {
        UNIT_ASSERT(it.Valid());
        UNIT_ASSERT(bool(it));
        UNIT_ASSERT_STRINGS_EQUAL(it->ToString(), expectedString);
        ++it;
    }
    UNIT_ASSERT(!it.Valid());
};

SIMPLE_UNIT_TEST_SUITE(TDelimStrokaIterTestSuite) {
    SIMPLE_UNIT_TEST(SingleCharacterAsDelimiter) {
        AssertStringSplit(
            "Hello words!", " ", {"Hello", "words!"});
    }

    SIMPLE_UNIT_TEST(MultipleCharactersAsDelimiter) {
        AssertStringSplit(
            "0, 1, 1, 2, 3, 5, 8, 13, 21, 34", "1, ", {"0, ", "", "2, 3, 5, 8, 13, 2", "34"});
    }

    SIMPLE_UNIT_TEST(NoDelimitersPresent) {
        AssertStringSplit("This string could be yours", "\t", {"This string could be yours"});
    }

    SIMPLE_UNIT_TEST(Cdr) {
        TDelimStrokaIter it("a\tc\t", "\t");
        UNIT_ASSERT_STRINGS_EQUAL(*it, "a");
        UNIT_ASSERT_STRINGS_EQUAL(it.Cdr(), "c\t");
        ++it;
        UNIT_ASSERT_STRINGS_EQUAL(it.Cdr(), "");
    }

    SIMPLE_UNIT_TEST(ForIter) {
        yvector<TStringBuf> expected = {"1", "", "3@4", ""};
        yvector<TStringBuf> got;

        for (TStringBuf x : TDelimStroka("1@@@@3@4@@", "@@")) {
            got.push_back(x);
        }

        UNIT_ASSERT_EQUAL(got, expected);
    }
}

static void AssertKeyValueStringSplit(
    const TStringBuf str,
    const TStringBuf delim,
    const yvector<std::pair<TStringBuf, TStringBuf>>& expected) {
    TKeyValueDelimStrokaIter it(str, delim);

    for (const auto& expectedKeyValue : expected) {
        UNIT_ASSERT(it.Valid());
        UNIT_ASSERT_STRINGS_EQUAL(it.Key(), expectedKeyValue.first);
        UNIT_ASSERT_STRINGS_EQUAL(it.Value(), expectedKeyValue.second);
        ++it;
    }
    UNIT_ASSERT(!it.Valid());
}

SIMPLE_UNIT_TEST_SUITE(TKeyValueDelimStrokaIterTestSuite) {
    SIMPLE_UNIT_TEST(SingleCharacterAsDelimiter) {
        AssertKeyValueStringSplit(
            "abc=123,cde=qwer", ",",
            {{"abc", "123"},
             {"cde", "qwer"}});
    }

    SIMPLE_UNIT_TEST(MultipleCharactersAsDelimiter) {
        AssertKeyValueStringSplit(
            "abc=xyz@@qwerty=zxcv", "@@",
            {{"abc", "xyz"},
             {"qwerty", "zxcv"}});
    }

    SIMPLE_UNIT_TEST(NoDelimiters) {
        AssertKeyValueStringSplit(
            "abc=zz", ",",
            {{"abc", "zz"}});
    }

    SIMPLE_UNIT_TEST(EmptyElements) {
        AssertKeyValueStringSplit(
            "@@abc=zxy@@@@qwerty=y@@", "@@",
            {{"", ""},
             {"abc", "zxy"},
             {"", ""},
             {"qwerty", "y"},
             {"", ""}});
    }
}
