#include "split.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>
#include <util/charset/wide.h>
#include <util/datetime/cputimer.h>
#include <util/generic/maybe.h>

#include <string>
#include <string_view>

template <typename T>
static inline void OldSplit(char* pszBuf, T* pRes) {
    pRes->resize(0);
    pRes->push_back(pszBuf);
    for (char* pszData = pszBuf; *pszData; ++pszData) {
        if (*pszData == '\t') {
            *pszData = 0;
            pRes->push_back(pszData + 1);
        }
    }
}

template <class T1, class T2>
inline void Cmp(const T1& t1, const T2& t2) {
    try {
        UNIT_ASSERT_EQUAL(t1.size(), t2.size());
    } catch (...) {
        Print(t1);
        Cerr << "---------------" << Endl;
        Print(t2);

        throw;
    }

    auto i = t1.begin();
    auto j = t2.begin();

    for (; i != t1.end() && j != t2.end(); ++i, ++j) {
        try {
            UNIT_ASSERT_EQUAL(*i, *j);
        } catch (...) {
            Cerr << "(" << *i << ")->(" << *j << ")" << Endl;

            throw;
        }
    }
}

template <class T>
inline void Print(const T& t) {
    for (typename T::const_iterator i = t.begin(); i != t.end(); ++i) {
        Cerr << *i << Endl;
    }
}

template <template <typename> class TConsumer, typename TResult, typename I, typename TDelimiter>
void TestDelimiterOnString(TResult& good, I* str, const TDelimiter& delim) {
    TResult test;
    TConsumer<TResult> consumer(&test);
    SplitString(str, delim, consumer);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
}

template <template <typename> class TConsumer, typename TResult, typename I, typename TDelimiter>
void TestDelimiterOnRange(TResult& good, I* b, I* e, const TDelimiter& delim) {
    TResult test;
    TConsumer<TResult> consumer(&test);
    SplitString(b, e, delim, consumer);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
}

template <typename TConsumer, typename TResult, typename I>
void TestConsumerOnString(TResult& good, I* str, I* d) {
    TResult test;
    TContainerConsumer<TResult> consumer(&test);
    TConsumer tested(&consumer);
    TCharDelimiter<const I> delim(*d);
    SplitString(str, delim, tested);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
}

template <typename TConsumer, typename TResult, typename I>
void TestConsumerOnRange(TResult& good, I* b, I* e, I* d) {
    TResult test;
    TContainerConsumer<TResult> consumer(&test);
    TConsumer tested(&consumer);
    TCharDelimiter<const I> delim(*d);
    SplitString(b, e, delim, tested);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
}

using TStrokaConsumer = TContainerConsumer<TVector<TString>>;

void TestLimitingConsumerOnString(TVector<TString>& good, const char* str, const char* d, size_t n, const char* last) {
    TVector<TString> test;
    TStrokaConsumer consumer(&test);
    TLimitingConsumer<TStrokaConsumer, const char> limits(n, &consumer);
    TCharDelimiter<const char> delim(*d);
    SplitString(str, delim, limits);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
    UNIT_ASSERT_EQUAL(TString(limits.Last), TString(last)); // Quite unobvious behaviour. Why the last token is not added to slave consumer?
}

void TestLimitingConsumerOnRange(TVector<TString>& good, const char* b, const char* e, const char* d, size_t n, const char* last) {
    TVector<TString> test;
    TStrokaConsumer consumer(&test);
    TLimitingConsumer<TStrokaConsumer, const char> limits(n, &consumer);
    TCharDelimiter<const char> delim(*d);
    SplitString(b, e, delim, limits);
    Cmp(good, test);
    UNIT_ASSERT_EQUAL(good, test);
    UNIT_ASSERT_EQUAL(TString(limits.Last), TString(last));
}

Y_UNIT_TEST_SUITE(SplitStringTest) {
    Y_UNIT_TEST(TestCharSingleDelimiter) {
        TString data("qw ab  qwabcab");
        TString canonic[] = {"qw", "ab", "", "qwabcab"};
        TVector<TString> good(canonic, canonic + 4);
        TCharDelimiter<const char> delim(' ');

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestWideSingleDelimiter) {
        TUtf16String data(u"qw ab  qwabcab");
        TUtf16String canonic[] = {u"qw", u"ab", TUtf16String(), u"qwabcab"};
        TVector<TUtf16String> good(canonic, canonic + 4);
        TCharDelimiter<const wchar16> delim(' ');

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestConvertToIntCharSingleDelimiter) {
        TString data("42 4242 -12345 0");
        i32 canonic[] = {42, 4242, -12345, 0};
        TVector<i32> good(canonic, canonic + 4);
        TCharDelimiter<const char> delim(' ');

        TestDelimiterOnString<TContainerConvertingConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConvertingConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestCharSkipEmpty) {
        TString data("qw ab  qwabcab ");
        TString canonic[] = {"qw", "ab", "qwabcab"};
        TVector<TString> good(canonic, canonic + 3);

        TestConsumerOnString<TSkipEmptyTokens<TStrokaConsumer>>(good, data.data(), " ");
        TestConsumerOnRange<TSkipEmptyTokens<TStrokaConsumer>>(good, data.data(), data.end(), " ");
    }

    Y_UNIT_TEST(TestCharKeepDelimiters) {
        TString data("qw ab  qwabcab ");
        TString canonic[] = {"qw", " ", "ab", " ", "", " ", "qwabcab", " ", ""};
        TVector<TString> good(canonic, canonic + 9);

        TestConsumerOnString<TKeepDelimiters<TStrokaConsumer>>(good, data.data(), " ");
        TestConsumerOnRange<TKeepDelimiters<TStrokaConsumer>>(good, data.data(), data.end(), " ");
    }

    Y_UNIT_TEST(TestCharLimit) {
        TString data("qw ab  qwabcab ");
        TString canonic[] = {"qw", "ab"};
        TVector<TString> good(canonic, canonic + 2);

        TestLimitingConsumerOnString(good, data.data(), " ", 3, " qwabcab ");
        TestLimitingConsumerOnRange(good, data.data(), data.end(), " ", 3, " qwabcab ");
    }

    Y_UNIT_TEST(TestCharStringDelimiter) {
        TString data("qw ab qwababcab");
        TString canonic[] = {"qw ", " qw", "", "c", ""};
        TVector<TString> good(canonic, canonic + 5);
        TStringDelimiter<const char> delim("ab");

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestWideStringDelimiter) {
        TUtf16String data(u"qw ab qwababcab");
        TUtf16String canonic[] = {u"qw ", u" qw", TUtf16String(), u"c", TUtf16String()};
        TVector<TUtf16String> good(canonic, canonic + 5);
        TUtf16String wideDelim(u"ab");
        TStringDelimiter<const wchar16> delim(wideDelim.data());

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestCharSetDelimiter) {
        TString data("qw ab qwababccab");
        TString canonic[] = {"q", " ab q", "abab", "", "ab"};
        TVector<TString> good(canonic, canonic + 5);
        TSetDelimiter<const char> delim("wc");

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
        TestDelimiterOnRange<TContainerConsumer>(good, data.data(), data.end(), delim);
    }

    Y_UNIT_TEST(TestWideSetDelimiter) {
        TUtf16String data(u"qw ab qwababccab");
        TUtf16String canonic[] = {u"q", u" ab q", u"abab", TUtf16String(), u"ab"};
        TVector<TUtf16String> good(canonic, canonic + 5);
        TUtf16String wideDelim(u"wc");
        TSetDelimiter<const wchar16> delim(wideDelim.data());

        TestDelimiterOnString<TContainerConsumer>(good, data.data(), delim);
    }

    Y_UNIT_TEST(TestWideSetDelimiterRange) {
        TUtf16String data(u"qw ab qwababccab");
        TUtf16String canonic[] = {u"q", u" ab q", u"abab", TUtf16String(), u"ab"};
        TVector<TUtf16String> good(1);
        TUtf16String wideDelim(u"wc");
        TSetDelimiter<const wchar16> delim(wideDelim.data());

        TVector<TUtf16String> test;
        TContainerConsumer<TVector<TUtf16String>> consumer(&test);
        SplitString(data.data(), data.data(), delim, consumer); // Empty string is still inserted into consumer
        Cmp(good, test);

        good.assign(canonic, canonic + 4);
        good.push_back(TUtf16String());
        test.clear();
        SplitString(data.data(), data.end() - 2, delim, consumer);
        Cmp(good, test);
    }

    Y_UNIT_TEST(TestSplit) {
        TString data("qw ab qwababcba");
        TString canonic[] = {"qw ", " qw", "c"};
        TVector<TString> good(canonic, canonic + 3);
        TString delim = "ab";
        TVector<TString> test;
        Split(data, delim, test);
        Cmp(good, test);

        TVector<TStringBuf> test1;
        Split(data, delim.data(), test1);
        Cmp(good, test1);
    }

    Y_UNIT_TEST(ConvenientSplitTest) {
        TString data("abc 22 33.5 xyz");
        TString str;
        int num1 = 0;
        double num2 = 0;
        TStringBuf strBuf;
        Split(data, ' ', str, num1, num2, strBuf);
        UNIT_ASSERT_VALUES_EQUAL(str, "abc");
        UNIT_ASSERT_VALUES_EQUAL(num1, 22);
        UNIT_ASSERT_VALUES_EQUAL(num2, 33.5);
        UNIT_ASSERT_VALUES_EQUAL(strBuf, "xyz");
    }

    Y_UNIT_TEST(ConvenientSplitTestWithMaybe) {
        TString data("abc 42");
        TString str;
        TMaybe<double> num2 = 1;
        TMaybe<double> maybe = 1;

        Split(data, ' ', str, num2, maybe);

        UNIT_ASSERT_VALUES_EQUAL(str, "abc");
        UNIT_ASSERT_VALUES_EQUAL(*num2, 42);
        UNIT_ASSERT(!maybe);
    }

    Y_UNIT_TEST(ConvenientSplitTestExceptions) {
        TString data("abc 22 33");
        TString s1, s2, s3, s4;

        UNIT_ASSERT_EXCEPTION(Split(data, ' ', s1, s2), yexception);
        UNIT_ASSERT_NO_EXCEPTION(Split(data, ' ', s1, s2, s3));
        UNIT_ASSERT_EXCEPTION(Split(data, ' ', s1, s2, s3, s4), yexception);
    }

    Y_UNIT_TEST(ConvenientSplitTestMaybeExceptions) {
        TString data("abc 22 33");
        TString s1, s2;
        TMaybe<TString> m1, m2;

        UNIT_ASSERT_EXCEPTION(Split(data, ' ', s1, m1), yexception);
        UNIT_ASSERT_EXCEPTION(Split(data, ' ', m1, m2), yexception);
        UNIT_ASSERT_NO_EXCEPTION(Split(data, ' ', s1, s2, m1));

        UNIT_ASSERT_NO_EXCEPTION(Split(data, ' ', s1, s2, m1, m2));
        UNIT_ASSERT_EXCEPTION(Split(data, ' ', m1, m2, s1, s2), yexception);

        UNIT_ASSERT_NO_EXCEPTION(Split(data, ' ', s1, s2, m1, m2, m1, m1, m1, m1));
        UNIT_ASSERT_EXCEPTION(Split(data, ' ', s1, s2, m1, m2, m1, m1, m1, m1, s1), yexception);
    }
}

template <typename I, typename C>
void TestStringSplitterCount(I* str, C delim, size_t good) {
    auto split = StringSplitter(str).Split(delim);
    size_t res = split.Count();
    UNIT_ASSERT_VALUES_EQUAL(res, good);
    res = split.Count();
    UNIT_ASSERT_VALUES_EQUAL(res, 0);
}

Y_UNIT_TEST_SUITE(StringSplitter) {
    Y_UNIT_TEST(TestSplit) {
        int sum = 0;

        for (const auto& it : StringSplitter("1,2,3").Split(',')) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestSplit1) {
        int cnt = 0;

        for (const auto& it : StringSplitter(" ").Split(' ')) {
            (void)it;

            ++cnt;
        }

        UNIT_ASSERT_VALUES_EQUAL(cnt, 2);
    }

    Y_UNIT_TEST(TestSplitLimited) {
        TVector<TString> expected = {"1", "2", "3,4,5"};
        TVector<TString> actual = StringSplitter("1,2,3,4,5").Split(',').Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitLimitedWithEmptySkip) {
        TVector<TString> expected = {"1", "2", "3,4,5"};
        TVector<TString> actual = StringSplitter("1,,,2,,,,3,4,5").Split(',').SkipEmpty().Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);

        expected = {"1", "2", ",,,3,4,5"};
        actual = StringSplitter("1,2,,,,3,4,5").Split(',').Limit(3).SkipEmpty().ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitBySet) {
        int sum = 0;

        for (const auto& it : StringSplitter("1,2:3").SplitBySet(",:")) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestSplitBySetLimited) {
        TVector<TString> expected = {"1", "2", "3,4:5"};
        TVector<TString> actual = StringSplitter("1,2:3,4:5").SplitBySet(",:").Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitBySetLimitedWithEmptySkip) {
        TVector<TString> expected = {"1", "2", "3,4:5"};
        TVector<TString> actual = StringSplitter("1,:,2::::,3,4:5").SplitBySet(",:").SkipEmpty().Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);

        expected = {"1", ",2::::,3,4:5"};
        actual = StringSplitter("1,:,2::::,3,4:5").SplitBySet(",:").Limit(3).SkipEmpty().ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitByString) {
        int sum = 0;

        for (const auto& it : StringSplitter("1ab2ab3").SplitByString("ab")) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestSplitByStringLimited) {
        TVector<TString> expected = {"1", "2", "3ab4ab5"};
        TVector<TString> actual = StringSplitter("1ab2ab3ab4ab5").SplitByString("ab").Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitByStringLimitedWithEmptySkip) {
        TVector<TString> expected = {"1", "2", "3ab4ab5"};
        TVector<TString> actual = StringSplitter("1abab2ababababab3ab4ab5").SplitByString("ab").SkipEmpty().Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitByFunc) {
        TString s = "123 456 \t\n789\n10\t 20";
        TVector<TString> pattern = {"123", "456", "789", "10", "20"};

        TVector<TString> tokens;
        auto f = [](char a) { return a == ' ' || a == '\t' || a == '\n'; };
        for (auto v : StringSplitter(s).SplitByFunc(f)) {
            if (v) {
                tokens.emplace_back(v);
            }
        }

        UNIT_ASSERT(tokens == pattern);
    }

    Y_UNIT_TEST(TestSplitByFuncLimited) {
        TVector<TString> expected = {"1", "2", "3a4b5"};
        auto f = [](char a) { return a == 'a' || a == 'b'; };
        TVector<TString> actual = StringSplitter("1a2b3a4b5").SplitByFunc(f).Limit(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSplitByFuncLimitedWithEmptySkip) {
        TVector<TString> expected = {"1", "2", "3a4b5"};
        auto f = [](char a) { return a == 'a' || a == 'b'; };
        TVector<TString> actual = StringSplitter("1aaba2bbababa3a4b5").SplitByFunc(f).SkipEmpty().Limit(3).Take(3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestSkipEmpty) {
        int sum = 0;

        for (const auto& it : StringSplitter("  1 2 3   ").Split(' ').SkipEmpty()) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);

        // double
        sum = 0;
        for (const auto& it : StringSplitter("  1 2 3   ").Split(' ').SkipEmpty().SkipEmpty()) {
            sum += FromString<int>(it.Token());
        }
        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestTake) {
        TVector<TString> expected = {"1", "2", "3"};
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("1 2 3 4 5 6 7 8 9 10").Split(' ').Take(3).ToList<TString>());

        expected = {"1", "2"};
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("  1 2 3   ").Split(' ').SkipEmpty().Take(2).ToList<TString>());

        expected = {"1", "2", "3"};
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("1 2 3 4 5 6 7 8 9 10").Split(' ').Take(5).Take(3).ToList<TString>());
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("1 2 3 4 5 6 7 8 9 10").Split(' ').Take(3).Take(5).ToList<TString>());

        expected = {"1", "2"};
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("  1 2 3  ").Split(' ').Take(4).SkipEmpty().ToList<TString>());

        expected = {"1"};
        UNIT_ASSERT_VALUES_EQUAL(expected, StringSplitter("  1 2 3  ").Split(' ').Take(4).SkipEmpty().Take(1).ToList<TString>());
    }

    Y_UNIT_TEST(TestCompile) {
        (void)StringSplitter(TString());
        (void)StringSplitter(TStringBuf());
        (void)StringSplitter("", 0);
    }

    Y_UNIT_TEST(TestStringSplitterCountEmpty) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("", delim, 1);
    }

    Y_UNIT_TEST(TestStringSplitterCountOne) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("one", delim, 1);
    }

    Y_UNIT_TEST(TestStringSplitterCountWithOneDelimiter) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("one two", delim, 2);
    }

    Y_UNIT_TEST(TestStringSplitterCountWithTrailing) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount(" one ", delim, 3);
    }

    Y_UNIT_TEST(TestStringSplitterConsume) {
        TVector<TString> expected = {"1", "2", "3"};
        TVector<TString> actual;
        auto func = [&actual](const TBasicStringBuf<char>& token) {
            actual.push_back(TString(token));
        };
        StringSplitter("1 2 3").Split(' ').Consume(func);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStringSplitterConsumeConditional) {
        TVector<TString> expected = {"1", "2"};
        TVector<TString> actual;
        auto func = [&actual](const TBasicStringBuf<char>& token) {
            if (token == "3") {
                return false;
            }
            actual.push_back(TString(token));
            return true;
        };
        bool completed = StringSplitter("1 2 3 4 5").Split(' ').Consume(func);
        UNIT_ASSERT(!completed);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStringSplitterToList) {
        TVector<TString> expected = {"1", "2", "3"};
        TVector<TString> actual = StringSplitter("1 2 3").Split(' ').ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStringSplitterCollectPushBack) {
        TVector<TString> expected = {"1", "2", "3"};
        TVector<TString> actual;
        StringSplitter("1 2 3").Split(' ').Collect(&actual);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStringSplitterCollectInsert) {
        TSet<TString> expected = {"1", "2", "3"};
        TSet<TString> actual;
        StringSplitter("1 2 3 1 2 3").Split(' ').Collect(&actual);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStringSplitterCollectClears) {
        TVector<TString> v;
        StringSplitter("1 2 3").Split(' ').Collect(&v);
        UNIT_ASSERT_VALUES_EQUAL(v.size(), 3);
        StringSplitter("4 5").Split(' ').Collect(&v);
        UNIT_ASSERT_VALUES_EQUAL(v.size(), 2);
    }

    Y_UNIT_TEST(TestStringSplitterAddToDoesntClear) {
        TVector<TString> v;
        StringSplitter("1 2 3").Split(' ').AddTo(&v);
        UNIT_ASSERT_VALUES_EQUAL(v.size(), 3);
        StringSplitter("4 5").Split(' ').AddTo(&v);
        UNIT_ASSERT_VALUES_EQUAL(v.size(), 5);
    }

    Y_UNIT_TEST(TestSplitStringInto) {
        int a = -1;
        TStringBuf s;
        double d = -1;
        StringSplitter("2 substr 1.02").Split(' ').CollectInto(&a, &s, &d);
        UNIT_ASSERT_VALUES_EQUAL(a, 2);
        UNIT_ASSERT_VALUES_EQUAL(s, "substr");
        UNIT_ASSERT_DOUBLES_EQUAL(d, 1.02, 0.0001);
        UNIT_ASSERT_EXCEPTION(StringSplitter("1").Split(' ').CollectInto(&a, &a), yexception);
        UNIT_ASSERT_EXCEPTION(StringSplitter("1 2 3").Split(' ').CollectInto(&a, &a), yexception);
    }

    Y_UNIT_TEST(TestSplitStringWithIgnore) {
        TStringBuf s;
        StringSplitter("x y z").Split(' ').CollectInto(&std::ignore, &s, &std::ignore);
        UNIT_ASSERT_VALUES_EQUAL(s, "y");

        UNIT_ASSERT_EXCEPTION(StringSplitter("ignored != non-requred").Split(':').CollectInto(&s, &std::ignore), yexception);
    }

    Y_UNIT_TEST(TestTryCollectInto) {
        int a, b, c;
        bool parsingSucceeded;
        parsingSucceeded = StringSplitter("100,500,3").Split(',').TryCollectInto(&a, &b, &c);
        UNIT_ASSERT(parsingSucceeded);
        UNIT_ASSERT_VALUES_EQUAL(a, 100);
        UNIT_ASSERT_VALUES_EQUAL(b, 500);
        UNIT_ASSERT_VALUES_EQUAL(c, 3);

        // not enough tokens
        parsingSucceeded = StringSplitter("3,14").Split(',').TryCollectInto(&a, &b, &c);
        UNIT_ASSERT(!parsingSucceeded);

        // too many tokens
        parsingSucceeded = StringSplitter("3,14,15,92,6").Split(',').TryCollectInto(&a, &b, &c);
        UNIT_ASSERT(!parsingSucceeded);

        // where single TryFromString fails
        parsingSucceeded = StringSplitter("ot topota kopyt pyl po polu letit").Split(' ').TryCollectInto(&a, &b, &c);
        UNIT_ASSERT(!parsingSucceeded);
    }

    Y_UNIT_TEST(TestOwningSplit1) {
        int sum = 0;

        for (const auto& it : StringSplitter(TString("1,2,3")).Split(',')) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestOwningSplit2) {
        int sum = 0;

        TString str("1,2,3");
        for (const auto& it : StringSplitter(str).Split(',')) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestOwningSplit3) {
        int sum = 0;

        const TString str("1,2,3");
        for (const auto& it : StringSplitter(str).Split(',')) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    Y_UNIT_TEST(TestAssigment) {
        TVector<TString> expected0 = {"1", "2", "3", "4"};
        TVector<TString> actual0 = StringSplitter("1 2 3 4").Split(' ');
        UNIT_ASSERT_VALUES_EQUAL(expected0, actual0);

        TSet<TString> expected1 = {"11", "22", "33", "44"};
        TSet<TString> actual1 = StringSplitter("11 22 33 44").Split(' ');
        UNIT_ASSERT_VALUES_EQUAL(expected1, actual1);

        TSet<TString> expected2 = {"11", "aa"};
        auto actual2 = static_cast<TSet<TString>>(StringSplitter("11 aa 11 11 aa").Split(' '));
        UNIT_ASSERT_VALUES_EQUAL(expected2, actual2);

        TVector<TString> expected3 = {"dd", "bb"};
        auto actual3 = TVector<TString>(StringSplitter("dd\tbb").Split('\t'));
        UNIT_ASSERT_VALUES_EQUAL(expected3, actual3);
    }

    Y_UNIT_TEST(TestRangeBasedFor) {
        TVector<TString> actual0 = {"11", "22", "33", "44"};
        size_t num = 0;
        for (TStringBuf elem : StringSplitter("11 22 33 44").Split(' ')) {
            UNIT_ASSERT_VALUES_EQUAL(elem, actual0[num++]);
        }

        TVector<TString> actual1 = {"another", "one,", "and", "another", "one"};
        num = 0;
        for (TStringBuf elem : StringSplitter(TStringBuf("another one, and \n\n     another    one")).SplitBySet(" \n").SkipEmpty()) {
            UNIT_ASSERT_VALUES_EQUAL(elem, actual1[num++]);
        }

        TVector<TUtf16String> actual2 = {u"привет,", u"как", u"дела"};
        num = 0;
        for (TWtringBuf elem : StringSplitter(u"привет, как дела").Split(wchar16(' '))) {
            UNIT_ASSERT_VALUES_EQUAL(elem, actual2[num++]);
        }

        TVector<TString> copy(4);
        auto v = StringSplitter("11 22 33 44").Split(' ');
        Copy(v.begin(), v.end(), copy.begin());
        UNIT_ASSERT_VALUES_EQUAL(actual0, copy);
    }

    Y_UNIT_TEST(TestParseInto) {
        TVector<int> actual0 = {1, 2, 3, 4};
        TVector<int> answer0;

        StringSplitter("1 2 3 4").Split(' ').ParseInto(&answer0);
        UNIT_ASSERT_VALUES_EQUAL(actual0, answer0);

        TVector<int> actual1 = {42, 1, 2, 3, 4};
        TVector<int> answer1 = {42};
        StringSplitter("1 2 3 4").Split(' ').ParseInto(&answer1);
        UNIT_ASSERT_VALUES_EQUAL(actual1, answer1);

        answer1.clear();
        UNIT_ASSERT_EXCEPTION(StringSplitter("1 2    3 4").Split(' ').ParseInto(&answer1), yexception);

        answer1 = {42};
        StringSplitter("   1    2     3 4").Split(' ').SkipEmpty().ParseInto(&answer1);
        UNIT_ASSERT_VALUES_EQUAL(actual1, answer1);

        answer1.clear();
        StringSplitter("  \n 1    2  \n\n\n   3 4\n ").SplitBySet(" \n").SkipEmpty().ParseInto(&answer1);
        UNIT_ASSERT_VALUES_EQUAL(actual0, answer1);
    }

    Y_UNIT_TEST(TestStdString) {
        std::vector<std::string_view> r0, r1, answer = {"lol", "zomg"};
        std::string s = "lol zomg";
        for (std::string_view ss : StringSplitter(s).Split(' ')) {
            r0.push_back(ss);
        }
        StringSplitter(s).Split(' ').Collect(&r1);

        UNIT_ASSERT_VALUES_EQUAL(r0, answer);
        UNIT_ASSERT_VALUES_EQUAL(r1, answer);
    }

    Y_UNIT_TEST(TestStdStringView) {
        std::string_view s = "aaacccbbb";
        std::vector<std::string_view> expected = {"aaa", "bbb"};
        std::vector<std::string_view> actual = StringSplitter(s).SplitByString("ccc");
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    Y_UNIT_TEST(TestStdSplitAfterSplit) {
        std::string_view input = "a*b+a*b";
        for (std::string_view summand : StringSplitter(input).Split('+')) {
            // FIXME: std::string is used to workaround MSVC ICE
            UNIT_ASSERT_VALUES_EQUAL(std::string(summand), "a*b");
            std::string_view multiplier1, multiplier2;
            bool splitResult = StringSplitter(summand).Split('*').TryCollectInto(&multiplier1, &multiplier2);
            UNIT_ASSERT(splitResult);
            UNIT_ASSERT_VALUES_EQUAL(std::string(multiplier1), "a");
            UNIT_ASSERT_VALUES_EQUAL(std::string(multiplier2), "b");
        }
    }

    Y_UNIT_TEST(TestStdSplitWithParsing) {
        std::string_view input = "1,2,3,4";
        TVector<ui64> numbers;
        const TVector<ui64> expected{1, 2, 3, 4};
        StringSplitter(input).Split(',').ParseInto(&numbers);
        UNIT_ASSERT_VALUES_EQUAL(numbers, expected);
    }

    Y_UNIT_TEST(TestArcadiaStdInterop) {
        TVector<TString> expected0 = {"a", "b"};
        TVector<TStringBuf> expected1 = {"a", "b"};
        std::string src1("a  b");
        std::string_view src2("a  b");
        TVector<TString> actual0 = StringSplitter(src1).Split(' ').SkipEmpty();
        TVector<TString> actual1 = StringSplitter(src2).Split(' ').SkipEmpty();
        TVector<TStringBuf> actual2 = StringSplitter(src1).Split(' ').SkipEmpty();
        TVector<TStringBuf> actual3 = StringSplitter(src2).Split(' ').SkipEmpty();
        UNIT_ASSERT_VALUES_EQUAL(expected0, actual0);
        UNIT_ASSERT_VALUES_EQUAL(expected0, actual1);
        UNIT_ASSERT_VALUES_EQUAL(expected1, actual2);
        UNIT_ASSERT_VALUES_EQUAL(expected1, actual3);
    }

    Y_UNIT_TEST(TestConstCString) {
        const char* b = "a;b";
        const char* e = b + 3;

        std::vector<TStringBuf> v;
        StringSplitter(b, e).Split(';').AddTo(&v);

        std::vector<TStringBuf> expected = {"a", "b"};
        UNIT_ASSERT_VALUES_EQUAL(v, expected);
    }

    Y_UNIT_TEST(TestCStringRef) {
        TString s = "lol";
        char* str = s.Detach();

        std::vector<TStringBuf> v = StringSplitter(str).Split('o');
        std::vector<TStringBuf> expected = {"l", "l"};
        UNIT_ASSERT_VALUES_EQUAL(v, expected);
    }

    Y_UNIT_TEST(TestSplitVector) {
        std::vector<char> buffer = {'a', ';', 'b'};

        std::vector<TStringBuf> v = StringSplitter(buffer).Split(';');

        std::vector<TStringBuf> expected = {"a", "b"};
        UNIT_ASSERT_VALUES_EQUAL(v, expected);
    }

    class TDoubleIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = int;
        using pointer = void;
        using reference = int;
        using const_reference = int;
        using difference_type = ptrdiff_t;

        TDoubleIterator() = default;

        TDoubleIterator(const char* ptr)
            : Ptr_(ptr)
        {
        }

        TDoubleIterator operator++() {
            Ptr_ += 2;
            return *this;
        }

        TDoubleIterator operator++(int) {
            TDoubleIterator tmp = *this;
            ++*this;
            return tmp;
        }

        friend bool operator==(TDoubleIterator l, TDoubleIterator r) {
            return l.Ptr_ == r.Ptr_;
        }

        friend bool operator!=(TDoubleIterator l, TDoubleIterator r) {
            return l.Ptr_ != r.Ptr_;
        }

        int operator*() const {
            return (*Ptr_ - '0') * 10 + *(Ptr_ + 1) - '0';
        }

    private:
        const char* Ptr_ = nullptr;
    };

    Y_UNIT_TEST(TestInputIterator) {
        const char* beg = "1213002233000011";
        const char* end = beg + strlen(beg);

        std::vector<std::vector<int>> expected = {{12, 13}, {22, 33}, {}, {11}};
        int i = 0;

        for (TIteratorRange<TDoubleIterator> part : StringSplitter(TDoubleIterator(beg), TDoubleIterator(end)).SplitByFunc([](int value) { return value == 0; })) {
            UNIT_ASSERT(std::equal(part.begin(), part.end(), expected[i].begin(), expected[i].end()));
            i++;
        }
        UNIT_ASSERT_VALUES_EQUAL(i, expected.size());
    }
}
