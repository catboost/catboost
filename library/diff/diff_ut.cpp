#include "diff.h"

#include <library/unittest/registar.h>

using namespace NDiff;

struct TDiffTester {
    TStringStream Res;
    yvector<TChunk<char>> Chunks;

    TStringBuf Special(const TStringBuf& str) const {
        return str;
    }

    TStringBuf Common(const TConstArrayRef<const char>& str) const {
        return TStringBuf(str.begin(), str.end());
    }

    TStringBuf Left(const TConstArrayRef<const char>& str) const {
        return TStringBuf(str.begin(), str.end());
    }

    TStringBuf Right(const TConstArrayRef<const char>& str) const {
        return TStringBuf(str.begin(), str.end());
    }

    void Test(const TStringBuf& a, const TStringBuf& b, const TString& delims = " \t\n") {
        Chunks.clear();
        InlineDiff(Chunks, a, b, delims);
        Res.clear();
        PrintChunks(Res, *this, Chunks);
    }

    const TString& Result() const {
        return Res.Str();
    }
};

SIMPLE_UNIT_TEST_SUITE(DiffTokens) {
    SIMPLE_UNIT_TEST(EqualStringsOneToken) {
        TDiffTester tester;

        tester.Test("aaa", "aaa");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa");
    }

    SIMPLE_UNIT_TEST(NonCrossingStringsOneToken) {
        TDiffTester tester;

        tester.Test("aaa", "bbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|bbb)");

        tester.Test("aaa", "bbbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|bbbb)");
    }

    SIMPLE_UNIT_TEST(Simple) {
        TDiffTester tester;

        tester.Test("aaa", "abb", "");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "a(aa|bb)");

        tester.Test("aac", "abc", "");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "a(a|b)c");

        tester.Test("123", "133", "");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "1(2|3)3");

        tester.Test("[1, 2, 3]", "[1, 3, 3]", "");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "[1, (2|3), 3]");
    }

    SIMPLE_UNIT_TEST(CommonCharOneToken) {
        TDiffTester tester;

        tester.Test("abcde", "accfg");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(abcde|accfg)");
    }

    SIMPLE_UNIT_TEST(EqualStringsTwoTokens) {
        TDiffTester tester;

        TStringBuf str("aaa bbb");
        tester.Test(str, str);

        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa bbb");
    }

    SIMPLE_UNIT_TEST(NonCrossingStringsTwoTokens) {
        TDiffTester tester;

        tester.Test("aaa bbb", "ccc ddd");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|ccc) (bbb|ddd)");

        tester.Test("aaa bbb", "c d");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|c) (bbb|d)");
    }

    SIMPLE_UNIT_TEST(SimpleTwoTokens) {
        TDiffTester tester;

        tester.Test("aaa ccd", "abb cce");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|abb) (ccd|cce)");

        tester.Test("aac cbb", "aa bb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aac|aa) (cbb|bb)");
    }

    SIMPLE_UNIT_TEST(MixedTwoTokens) {
        TDiffTester tester;

        tester.Test("aaa bbb", "bbb aaa");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(|bbb )aaa( bbb|)");

        tester.Test("aaa bbb", " bbb aaa");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|) bbb(| aaa)");

        tester.Test(" aaa bbb ", " bbb aaa ");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(| bbb) aaa (bbb |)");

        tester.Test("aaa bb", " bbb aa");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|) (bb|bbb aa)");
    }

    SIMPLE_UNIT_TEST(TwoTokensInOneString) {
        TDiffTester tester;

        tester.Test("aaa bbb", "aaa");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa( bbb|)");

        tester.Test("aaa bbb", "aaa ");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa (bbb|)");

        tester.Test("aaa bbb", " bbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa|) bbb");

        tester.Test("aaa bbb", "bbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "(aaa |)bbb");
    }

    SIMPLE_UNIT_TEST(Multiline) {
        TDiffTester tester;

        tester.Test("aaa\nabc\nbbb", "aaa\nacc\nbbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa\n(abc|acc)\nbbb");

        tester.Test("aaa\nabc\nbbb", "aaa\nac\nbbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa\n(abc|ac)\nbbb");
    }

    SIMPLE_UNIT_TEST(DifferentDelimiters) {
        TDiffTester tester;

        tester.Test("aaa bbb", "aaa\tbbb");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "aaa( |\t)bbb");

        tester.Test(" aaa\tbbb\n", "\taaa\nbbb ");
        //~ Cerr << tester.Result() << Endl;
        UNIT_ASSERT_VALUES_EQUAL(tester.Result(), "( |\t)aaa(\t|\n)bbb(\n| )");
    }
}
