#include "iterator.h"

#include <library/unittest/registar.h>

template <typename I, typename C>
void TestStringSplitterCount(I* str, C delim, size_t good) {
    size_t res = StringSplitter(str).Split(delim).Count();
    UNIT_ASSERT_VALUES_EQUAL(res, good);
}

SIMPLE_UNIT_TEST_SUITE(StringSplitter) {
    SIMPLE_UNIT_TEST(TestSplit) {
        int sum = 0;

        for (const auto& it : StringSplitter("1,2,3").Split(',')) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    SIMPLE_UNIT_TEST(TestSplit1) {
        int cnt = 0;

        for (const auto& it : StringSplitter(" ").Split(' ')) {
            (void)it;

            ++cnt;
        }

        UNIT_ASSERT_VALUES_EQUAL(cnt, 2);
    }

    SIMPLE_UNIT_TEST(TestSplitLimited) {
        yvector<TString> expected = {"1", "2", "3,4,5"};
        yvector<TString> actual = StringSplitter("1,2,3,4,5").SplitLimited(',', 3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestSplitBySet) {
        int sum = 0;

        for (const auto& it : StringSplitter("1,2:3").SplitBySet(",:")) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    SIMPLE_UNIT_TEST(TestSplitBySetLimited) {
        yvector<TString> expected = {"1", "2", "3,4:5"};
        yvector<TString> actual = StringSplitter("1,2:3,4:5").SplitBySetLimited(",:", 3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestSplitByString) {
        int sum = 0;

        for (const auto& it : StringSplitter("1ab2ab3").SplitByString("ab")) {
            sum += FromString<int>(it.Token());
        }

        UNIT_ASSERT_VALUES_EQUAL(sum, 6);
    }

    SIMPLE_UNIT_TEST(TestSplitByStringLimited) {
        yvector<TString> expected = {"1", "2", "3ab4ab5"};
        yvector<TString> actual = StringSplitter("1ab2ab3ab4ab5").SplitByStringLimited("ab", 3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestSplitByFunc) {
        TString s = "123 456 \t\n789\n10\t 20";
        yvector<TString> pattern = {"123", "456", "789", "10", "20"};

        yvector<TString> tokens;
        auto f = [](char a) { return a == ' ' || a == '\t' || a == '\n'; };
        for (auto v : StringSplitter(s).SplitByFunc(f)) {
            if (v.Empty() == false) {
                tokens.emplace_back(v.TokenStart(), v.TokenDelim());
            }
        }

        UNIT_ASSERT(tokens == pattern);
    }

    SIMPLE_UNIT_TEST(TestSplitByFuncLimited) {
        yvector<TString> expected = {"1", "2", "3a4b5"};
        auto f = [](char a) { return a == 'a' || a == 'b'; };
        yvector<TString> actual = StringSplitter("1a2b3a4b5").SplitByFuncLimited(f, 3).ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestSkipEmpty) {
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

    SIMPLE_UNIT_TEST(TestTake) {
        yvector<TString> expected = {"1", "2", "3"};
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

    SIMPLE_UNIT_TEST(TestCompileAbility) {
        (void)StringSplitter(TString());
        (void)StringSplitter(TStringBuf());
        (void)StringSplitter("", 0);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCountEmpty) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("", delim, 1);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCountOne) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("one", delim, 1);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCountWithOneDelimiter) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount("one two", delim, 2);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCountWithTrailing) {
        TCharDelimiter<const char> delim(' ');
        TestStringSplitterCount(" one ", delim, 3);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterConsume) {
        yvector<TString> expected = {"1", "2", "3"};
        yvector<TString> actual;
        auto func = [&actual](const TGenericStringBuf<char>& token) {
            actual.push_back(TString(token));
        };
        StringSplitter("1 2 3").Split(' ').Consume(func);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterToList) {
        yvector<TString> expected = {"1", "2", "3"};
        yvector<TString> actual = StringSplitter("1 2 3").Split(' ').ToList<TString>();
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCollectPushBack) {
        yvector<TString> expected = {"1", "2", "3"};
        yvector<TString> actual;
        StringSplitter("1 2 3").Split(' ').Collect(&actual);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestStringSplitterCollectInsert) {
        yset<TString> expected = {"1", "2", "3"};
        yset<TString> actual;
        StringSplitter("1 2 3 1 2 3").Split(' ').Collect(&actual);
        UNIT_ASSERT_VALUES_EQUAL(expected, actual);
    }

    SIMPLE_UNIT_TEST(TestSplitStringInto) {
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
}
