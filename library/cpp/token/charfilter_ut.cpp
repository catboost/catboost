#include <fstream>
#include <iomanip>
#include <library/cpp/unittest/registar.h>

#include "charfilter.h"

namespace {
    class TTestFilter {
    public:
        bool Check(wchar16 c) const {
            return (c == '\'' || c == '`');
        }
    };

}

class TCharFilterTest: public TTestBase {
    UNIT_TEST_SUITE(TCharFilterTest);
    UNIT_TEST(TestFilterSubtokens);
    UNIT_TEST(TestNoResultSubtokens);
    UNIT_TEST(TestEmptyResultSubtokens);
    UNIT_TEST(TestTokenNoSubtokens);
    UNIT_TEST(TestNoChanges);
    UNIT_TEST_SUITE_END();

public:
    void TestFilterSubtokens();
    void TestNoResultSubtokens();
    void TestEmptyResultSubtokens();
    void TestTokenNoSubtokens();
    void TestNoChanges();
};

UNIT_TEST_SUITE_REGISTRATION(TCharFilterTest);

void TCharFilterTest::TestFilterSubtokens() {
    {
        //                              01234567890123456789012345678901234567890123456789012345678901234
        const TUtf16String text = u"`123'45''``'   '  '`ab'  c'`dd'``efg   hi`''gklm  '' `'  nopq'```";
        //                              ----------        -----  ------        ---------         --------
        //                                                             -----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 10);
        token.SubTokens.push_back(18, 5);
        token.SubTokens.push_back(25, 6);
        token.SubTokens.push_back(31, 5);
        token.SubTokens.push_back(39, 9);
        token.SubTokens.push_back(57, 8);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678901234567890123456789012345678901234
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "12345`'   '  ab  cddefg   higklm  '' `'  nopq");
        //                                                                        -----        --  ---      ------         ----
        //                                                                                            ---
        UNIT_ASSERT(result.SubTokens.size() == 6);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 5);
        UNIT_ASSERT(result.SubTokens[1].Pos == 13 && result.SubTokens[1].Len == 2);
        UNIT_ASSERT(result.SubTokens[2].Pos == 17 && result.SubTokens[2].Len == 3);
        UNIT_ASSERT(result.SubTokens[3].Pos == 20 && result.SubTokens[3].Len == 3);
        UNIT_ASSERT(result.SubTokens[4].Pos == 26 && result.SubTokens[4].Len == 6);
        UNIT_ASSERT(result.SubTokens[5].Pos == 41 && result.SubTokens[5].Len == 4);
    }
    {
        //                              0123456789012345678901234567890123456789012345678901234
        const TUtf16String text = u"`'''``'   '  '`ab'  c'`dd'``efg   hi`''gklm  '' `' '```";
        //                                            ---   ------- ---   --- -----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(14, 3);
        token.SubTokens.push_back(20, 7);
        token.SubTokens.push_back(28, 3);
        token.SubTokens.push_back(34, 3);
        token.SubTokens.push_back(38, 5);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678901234567890123456789012345678901234567
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "`'''``'   '  'ab'  cdd`efg   hi'gklm  '' `' '```");
        //                                                                                      --   --- ---   -- ----
        UNIT_ASSERT(result.SubTokens.size() == 5);
        UNIT_ASSERT(result.SubTokens[0].Pos == 14 && result.SubTokens[0].Len == 2);
        UNIT_ASSERT(result.SubTokens[1].Pos == 19 && result.SubTokens[1].Len == 3);
        UNIT_ASSERT(result.SubTokens[2].Pos == 23 && result.SubTokens[2].Len == 3);
        UNIT_ASSERT(result.SubTokens[3].Pos == 29 && result.SubTokens[3].Len == 2);
        UNIT_ASSERT(result.SubTokens[4].Pos == 32 && result.SubTokens[4].Len == 4);
    }
    {
        //                              01234567890123
        const TUtf16String text = u"a`bc'' 'def```";
        //                              -----  -----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 5);
        token.SubTokens.push_back(7, 5);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        0123456789
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc' def``");
        //                                                                        ---  ---
        UNIT_ASSERT(result.SubTokens.size() == 2);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 3);
        UNIT_ASSERT(result.SubTokens[1].Pos == 5 && result.SubTokens[1].Len == 3);
    }
    {
        //                              01234567890123
        const TUtf16String text = u"'''a`bc''``def";
        //                                -----   ----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 5);
        token.SubTokens.push_back(10, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        01234567890
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "''abc''`def");
        //                                                                          ---   ---
        UNIT_ASSERT(result.SubTokens.size() == 2);
        UNIT_ASSERT(result.SubTokens[0].Pos == 2 && result.SubTokens[0].Len == 3);
        UNIT_ASSERT(result.SubTokens[1].Pos == 8 && result.SubTokens[1].Len == 3);
    }
    {
        //                              01234567
        const TUtf16String text = u"abc-def`";
        //                              --- ----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 3);
        token.SubTokens.push_back(4, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        0123456
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc-def");
        //                                                                        --- ---
        UNIT_ASSERT(result.SubTokens.size() == 2);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 3);
        UNIT_ASSERT(result.SubTokens[1].Pos == 4 && result.SubTokens[1].Len == 3);
    }
}

void TCharFilterTest::TestNoResultSubtokens() {
    {
        //                              012345678901
        const TUtf16String text = u"'' ''''  '''";
        //                              -- ----  ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 2);
        token.SubTokens.push_back(3, 4);
        token.SubTokens.push_back(9, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "   ");
        UNIT_ASSERT(result.SubTokens.size() == 0);
    }
    {
        //                              256 characters - one subtoken
        const TUtf16String text = u"''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''";
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 256);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "");
        UNIT_ASSERT(result.SubTokens.size() == 0);
    }
    {
        //                              0123456789012345678901
        const TUtf16String text = u"    '' ''''  '''      ";
        //                                  -- ----  ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(4, 2);
        token.SubTokens.push_back(7, 4);
        token.SubTokens.push_back(13, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        0123456789012
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "             ");
        UNIT_ASSERT(result.SubTokens.size() == 0);
    }
    {
        //                              01234567890123456
        const TUtf16String text = u"'``''``''''`''```";
        //                                --   ---  ----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 2);
        token.SubTokens.push_back(7, 3);
        token.SubTokens.push_back(12, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        01234678
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "'`'``'``");
        UNIT_ASSERT(result.SubTokens.size() == 0);
    }
    {
        //                              0123456789
        const TUtf16String text = u"```''''``";
        //                              --   ----
        //                                ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 2);
        token.SubTokens.push_back(2, 3);
        token.SubTokens.push_back(5, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "");
        UNIT_ASSERT(result.SubTokens.size() == 0);
    }
}

void TCharFilterTest::TestEmptyResultSubtokens() {
    {
        //                              01234567890123456789
        const TUtf16String text = u"  a'' ''''  'b'c'   ";
        //                                --- ----  -----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 3);
        token.SubTokens.push_back(6, 4);
        token.SubTokens.push_back(12, 5);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        01234567890
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "  a   bc   ");
        UNIT_ASSERT(result.SubTokens.size() == 2);
        UNIT_ASSERT(result.SubTokens[0].Pos == 2 && result.SubTokens[0].Len == 1);
        UNIT_ASSERT(result.SubTokens[1].Pos == 6 && result.SubTokens[1].Len == 2);
    }
    {
        //                              01234567890123456789
        const TUtf16String text = u"  '' ''''  ''abc'   ";
        //                                -- ----  ------
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 2);
        token.SubTokens.push_back(5, 4);
        token.SubTokens.push_back(11, 6);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        01234567890
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "     abc   ");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 5 && result.SubTokens[0].Len == 3);
    }
    {
        //                              0123456789012345
        const TUtf16String text = u"  '' ''''  ''abc";
        //                                -- ----  -----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 2);
        token.SubTokens.push_back(5, 4);
        token.SubTokens.push_back(11, 5);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        01234567
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "     abc");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 5 && result.SubTokens[0].Len == 3);
    }
    {
        //                              01234567890123456789
        const TUtf16String text = u"abc'' ''''  '''   ";
        //                              ----- ----  ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 5);
        token.SubTokens.push_back(6, 4);
        token.SubTokens.push_back(12, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc      ");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 3);
    }
    {
        //                              012345678901234567
        const TUtf16String text = u"  'a' ''''  '''   ";
        //                                --- ----  ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 3);
        token.SubTokens.push_back(6, 4);
        token.SubTokens.push_back(12, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "  a      ");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 2 && result.SubTokens[0].Len == 1);
    }
    {
        //                              012345678901234567
        const TUtf16String text = u"'``''``'a''`''```";
        //                                --   ---  ----
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(2, 2);
        token.SubTokens.push_back(7, 3);
        token.SubTokens.push_back(12, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "'`'``a'``");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 5 && result.SubTokens[0].Len == 1);
    }
    {
        //                              0123456789
        const TUtf16String text = u"```a'''``";
        //                              --   ----
        //                                ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 2);
        token.SubTokens.push_back(2, 3);
        token.SubTokens.push_back(5, 4);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "a");
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 1);
    }
}

void TCharFilterTest::TestTokenNoSubtokens() {
    {
        const TUtf16String text = u"`123'45''``'   '  '`ab'  c'`dd'``efg   hi`''gklm  '' `'  nopq'```";
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "12345     ab  cddefg   higklm     nopq");
    }
    {
        const TUtf16String text = u"a'''''b```c3'''";
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc3");
    }
    {
        const TUtf16String text = u"```ab''''1``c''2";
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "ab1c2");
    }
}

void TCharFilterTest::TestNoChanges() {
    {
        //                              012
        const TUtf16String text = u"abc";
        //                              ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc");
        //                                                                        ---
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 3);
    }
    {
        //                              012345678
        const TUtf16String text = u"abc'de`f ";
        //                              --- -- -
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(0, 3);
        token.SubTokens.push_back(4, 2);
        token.SubTokens.push_back(7, 1);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        012345678
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "abc'de`f ");
        //                                                                        --- -- -
        UNIT_ASSERT(result.SubTokens.size() == 3);
        UNIT_ASSERT(result.SubTokens[0].Pos == 0 && result.SubTokens[0].Len == 3);
        UNIT_ASSERT(result.SubTokens[1].Pos == 4 && result.SubTokens[1].Len == 2);
        UNIT_ASSERT(result.SubTokens[2].Pos == 7 && result.SubTokens[2].Len == 1);
    }
    {
        //                              0123
        const TUtf16String text = u"`abc";
        //                               ---
        TWideToken token;
        token.Token = text.c_str();
        token.Leng = text.size();
        token.SubTokens.push_back(1, 3);

        TCharFilter<TTestFilter> filter(token.Leng);
        const TWideToken& result = filter.Filter(token);

        //                                                                        0123
        UNIT_ASSERT_STRINGS_EQUAL(WideToUTF8(result.Token, result.Leng).c_str(), "`abc");
        //                                                                         ---
        UNIT_ASSERT(result.SubTokens.size() == 1);
        UNIT_ASSERT(result.SubTokens[0].Pos == 1 && result.SubTokens[0].Len == 3);
    }
}
