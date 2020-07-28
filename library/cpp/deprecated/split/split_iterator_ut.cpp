#include "split_iterator.h"

#include <library/cpp/testing/unittest/registar.h>

class TSplitIteratorTest: public TTestBase {
    UNIT_TEST_SUITE(TSplitIteratorTest);
    UNIT_TEST(TestDelimiters);
    UNIT_TEST(TestDelimitersSplit);
    UNIT_TEST(TestDelimitersStrictSplit);
    UNIT_TEST(TestTail);
    UNIT_TEST(TestScreenedDelimitersSplit);
    UNIT_TEST(TestSubstringDelimiter);
    UNIT_TEST_SUITE_END();

public:
    void TestDelimiters();
    void TestDelimitersSplit();
    void TestDelimitersStrictSplit();
    void TestTail();
    void TestScreenedDelimitersSplit();
    void TestSubstringDelimiter();
};

void TSplitIteratorTest::TestDelimiters() {
    TSplitDelimiters delims("@");
    for (int i = 0; i < 256; ++i)
        if ('@' != i) {
            UNIT_ASSERT(!delims.IsDelimiter((ui8)i));
        } else {
            UNIT_ASSERT(delims.IsDelimiter((ui8)i));
        }
}

void TSplitIteratorTest::TestDelimitersSplit() {
    {
        TString s = "1a3b45cd";
        TSplitDelimiters delims("abcd");
        TDelimitersSplit split(s, delims);
        TSplitTokens tokens;
        Split(split, &tokens);
        TSplitTokens pattern = {"1", "3", "45"};
        UNIT_ASSERT(tokens == pattern);
    }
    {
        TString s = "aaaaaa";
        TSplitDelimiters delims("abcd");
        TDelimitersSplit split(s, delims);
        TSplitTokens tokens;
        Split(split, &tokens);
        TSplitTokens pattern = {};
        UNIT_ASSERT(tokens == pattern);
    }
}

void TSplitIteratorTest::TestDelimitersStrictSplit() {
    {
        TString s = "grp@2";
        TSplitDelimiters delims("@");
        TDelimitersStrictSplit split(s, delims);
        TSplitTokens tokens;
        Split(split, &tokens);
        TSplitTokens pattern = {"grp", "2"};
        UNIT_ASSERT(tokens == pattern);
    }

    {
        TString s = "@grp@2@@";
        TSplitDelimiters delims("@");
        TDelimitersStrictSplit split(s, delims);
        TSplitTokens tokens;
        Split(split, &tokens);
        TSplitTokens pattern = {"", "grp", "2", ""};
        UNIT_ASSERT(tokens == pattern);
    }
}

void TSplitIteratorTest::TestTail() {
    TString s = "grp@2@4";
    TSplitDelimiters delims("@");
    TDelimitersSplit split(s, delims);
    TDelimitersSplit::TIterator it = split.Iterator();
    UNIT_ASSERT_EQUAL(it.GetTail(), "grp@2@4");
    it.Next();
    UNIT_ASSERT_EQUAL(it.GetTail(), "2@4");
    it.Next();
    UNIT_ASSERT_EQUAL(it.GetTail(), "4");
    it.Next();
    UNIT_ASSERT_EQUAL(it.GetTail(), "");
}

void TSplitIteratorTest::TestScreenedDelimitersSplit() {
    {
        const TString s = "77.88.58.91 - - [28/Aug/2008:00:08:07 +0400] \"GET /export/mordashka.tgz HTTP/1.1\" 304 - \"-\" \"libwww-perl/5.805\" \"news.yandex.ru,80\" \"-\" \"-\" 1219867687 \"0\" 3283 2";
        const TSplitDelimiters delims(" ");
        const TSplitDelimiters screens("\"[]");
        const TScreenedDelimitersSplit splitter(s, delims, screens);
        TScreenedDelimitersSplit::TIterator it = splitter.Iterator();
        UNIT_ASSERT_EQUAL(it.NextString(), "77.88.58.91");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "[28/Aug/2008:00:08:07 +0400]");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"GET /export/mordashka.tgz HTTP/1.1\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "304");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"libwww-perl/5.805\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"news.yandex.ru,80\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "1219867687");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"0\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "3283");
        UNIT_ASSERT_EQUAL(it.NextString(), "2");
    }
    {
        const TString s = "77.88.58.91 - - [28/Aug/2008:00:08:07 +0400] \"GET /export/mordashka.tgz HTTP/1.1\" 304 - \"-\" \"libwww-perl/5.805\" \"news.yandex.ru,80\" \"-\" \"-\" 1219867687 \"0\" 3283 2";
        const TSplitDelimiters delims(" ");
        const TSplitDelimiters screens("\"[]");
        const TScreenedDelimitersSplit splitter(s.Data(), s.Size(), delims, screens);
        TScreenedDelimitersSplit::TIterator it = splitter.Iterator();
        UNIT_ASSERT_EQUAL(it.NextString(), "77.88.58.91");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "[28/Aug/2008:00:08:07 +0400]");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"GET /export/mordashka.tgz HTTP/1.1\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "304");
        UNIT_ASSERT_EQUAL(it.NextString(), "-");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"libwww-perl/5.805\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"news.yandex.ru,80\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"-\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "1219867687");
        UNIT_ASSERT_EQUAL(it.NextString(), "\"0\"");
        UNIT_ASSERT_EQUAL(it.NextString(), "3283");
        UNIT_ASSERT_EQUAL(it.NextString(), "2");
    }
}

void TSplitIteratorTest::TestSubstringDelimiter() {
    const TString s = "a@@bb@@cc@c.d@@r";
    static const TSubstringSplitDelimiter delimiter("@@");
    const TSubstringSplit splitter(s, delimiter);
    TSubstringSplit::TIterator it = splitter.Iterator();
    UNIT_ASSERT_EQUAL(it.NextString(), "a");
    UNIT_ASSERT_EQUAL(it.NextString(), "bb");
    UNIT_ASSERT_EQUAL(it.NextString(), "cc@c.d");
    UNIT_ASSERT_EQUAL(it.NextString(), "r");
    UNIT_ASSERT(it.Eof());
}

UNIT_TEST_SUITE_REGISTRATION(TSplitIteratorTest);
