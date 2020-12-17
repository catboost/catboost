#include "chartraits.h"

#include <library/cpp/testing/unittest/registar.h>
#include <util/charset/unidata.h>

Y_UNIT_TEST_SUITE(TCharTraits) {
    Y_UNIT_TEST(TestLength) {
        using T = TCharTraits<char>;
        UNIT_ASSERT_EQUAL(T::GetLength("", 0), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 0), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 1), 1);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 3), 3);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 4), 3);
        UNIT_ASSERT_EQUAL(T::GetLength("abc", 1000), 3);

        // '\0'
        UNIT_ASSERT_EQUAL(T::GetLength("\0", 1000), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("\0abc", 1000), 0);
        UNIT_ASSERT_EQUAL(T::GetLength("a\0bc", 1000), 1);
    }

    Y_UNIT_TEST(TestCompare) {
        using T = TCharTraits<char>;
        // single char
        UNIT_ASSERT(T::Compare('a', 'a') == 0);
        UNIT_ASSERT(T::Compare('a', 'b') < 0);
        UNIT_ASSERT(T::Compare('b', 'a') > 0);

        // empty
        UNIT_ASSERT(T::Compare(nullptr, nullptr, 0) == 0);
        UNIT_ASSERT(T::Compare("", "") == 0);
        UNIT_ASSERT(T::Compare("", "", 0) == 0);
        UNIT_ASSERT(T::Compare("", "a") < 0);
        UNIT_ASSERT(T::Compare("abcd", "") > 0);

        UNIT_ASSERT(T::Compare("abc", "abc") == 0);
        UNIT_ASSERT(T::Compare("abcd", "abc") > 0);
        UNIT_ASSERT(T::Compare("ab", "abc") < 0);
        UNIT_ASSERT(T::Compare("abcd", "bcda") < 0);
        UNIT_ASSERT(T::Compare("bcd", "abcd") > 0);

        UNIT_ASSERT(T::Compare("ab", "abc", 2) == 0);
        UNIT_ASSERT(T::Compare("abcA", "abcB", 3) == 0);
        UNIT_ASSERT(T::Compare("abcA", "abcB", 4) < 0);

        // '\0' in the middle
        UNIT_ASSERT(T::Compare("ab\0ab", "ab\0ab", 5) == 0);
        UNIT_ASSERT(T::Compare("ab\0ab", "ab\0cd", 5) < 0);
        UNIT_ASSERT(T::Compare("ab\0cd", "ab\0ab", 5) > 0);
        UNIT_ASSERT(T::Compare("ab\0ab", "ab\0cd") == 0);
        UNIT_ASSERT(T::Compare("ab\0cd", "ab\0ab") == 0);
    }

    Y_UNIT_TEST(TestLongCompare) {
        using T = TCharTraits<char>;

        i8 data1[Max<ui8>()];
        i8 data2[Max<ui8>()];

        for (ui8 i = 0; i < Max<ui8>(); ++i) {
            data1[i] = i;
            data2[i] = i;
        }

        UNIT_ASSERT(T::Compare(reinterpret_cast<const char*>(&data1[0]), reinterpret_cast<const char*>(&data2[0]), sizeof(data1)) == 0);
        UNIT_ASSERT(T::Compare(reinterpret_cast<const char*>(&data1[0]), reinterpret_cast<const char*>(&data2[1]), sizeof(data1) - 1) == -1);
        UNIT_ASSERT(T::Compare(reinterpret_cast<const char*>(&data1[1]), reinterpret_cast<const char*>(&data2[0]), sizeof(data1) - 1) == 1);
    }

    Y_UNIT_TEST(TestFind) {
        using T = TCharTraits<char>;

        const char* empty = "";
        const char* abba = "abba";
        UNIT_ASSERT_EQUAL(T::Find(empty, 'a'), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, 'a'), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 'b'), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, '*'), nullptr);

        UNIT_ASSERT_EQUAL(T::Find(abba, 'a', 1), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 'b', 1), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, 'b', 2), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, 'b', 3), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, '*', 3), nullptr);

        UNIT_ASSERT_EQUAL(T::Find(empty, ""), empty);
        UNIT_ASSERT_EQUAL(T::Find(empty, "a"), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, ""), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, "a"), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, "ab"), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, "abba"), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, "b"), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, "bb"), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, "ba"), abba + 2);
        UNIT_ASSERT_EQUAL(T::Find(abba, "abba*"), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, "*ab"), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, "ab\0***"), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, "\0**"), abba);

        UNIT_ASSERT_EQUAL(T::Find(empty, 0, "", 0), empty);
        UNIT_ASSERT_EQUAL(T::Find(empty, 0, "abba", 0), empty);
        UNIT_ASSERT_EQUAL(T::Find(empty, 0, "abba", 4), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(abba, 0, "", 0), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 0, "abba", 0), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 0, "abba", 4), nullptr);

        UNIT_ASSERT_EQUAL(T::Find(abba, 1, "abba", 0), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 1, "abba", 1), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 1, "abba", 4), nullptr);

        UNIT_ASSERT_EQUAL(T::Find(abba, 4, "abba", 0), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, "abba", 1), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, "abba", 4), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, "ba", 2), abba + 2);

        // self-search
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, abba, 1), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, abba, 4), abba);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, abba + 2, 2), abba + 2);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, abba + 2, 1), abba + 1);
        UNIT_ASSERT_EQUAL(T::Find(abba, 4, abba + 3, 1), abba);

        // '\0' in the middle
        const char* ba0bab = "ba\0bab";
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 'b'), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 'a'), ba0bab + 1);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, "ba"), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, "ba\0", 3), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 2, "ba\0", 3), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, "a\0ba"), ba0bab + 1);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, "a\0ba", 4), ba0bab + 1);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, "ba\0AAA"), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, "ba\0AAA", 3), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, "ba\0AAA", 4), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, "bab"), nullptr);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, "bab", 3), ba0bab + 3);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, ba0bab, 2), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, ba0bab, 3), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 6, ba0bab, 6), ba0bab);
        UNIT_ASSERT_EQUAL(T::Find(ba0bab, 5, ba0bab, 6), nullptr);
    }

    Y_UNIT_TEST(TestRFind) {
        using T = TCharTraits<char>;

        const char* empty = "";
        const char* abba = "abba";
        UNIT_ASSERT_EQUAL(T::RFind(empty, 'a'), nullptr);
        UNIT_ASSERT_EQUAL(T::RFind(abba, 'a'), abba + 3);
        UNIT_ASSERT_EQUAL(T::RFind(abba, 'b'), abba + 2);
        UNIT_ASSERT_EQUAL(T::RFind(abba, '*'), nullptr);

        UNIT_ASSERT_EQUAL(T::RFind(abba, 'a', 1), abba);
        UNIT_ASSERT_EQUAL(T::RFind(abba, 'b', 1), nullptr);
        UNIT_ASSERT_EQUAL(T::RFind(abba, 'b', 2), abba + 1);
        UNIT_ASSERT_EQUAL(T::RFind(abba, 'b', 3), abba + 2);
        UNIT_ASSERT_EQUAL(T::RFind(abba, '*', 3), nullptr);

        // TODO: tests for RFind(const TCharType*, size_t, const TCharType*, size_t, size_t)
    }

    template <typename TCharType>
    void TestToLower() {
        using T = TCharTraits<TCharType>;
        UNIT_ASSERT_EQUAL(T::ToLower('A'), 'a');
        UNIT_ASSERT_EQUAL(T::ToLower('a'), 'a');
        UNIT_ASSERT_EQUAL(T::ToLower(' '), ' ');
        UNIT_ASSERT_EQUAL(T::ToLower('4'), '4');
        UNIT_ASSERT_EQUAL(T::ToLower('$'), '$');
    }

    Y_UNIT_TEST(TestCase) {
        TestToLower<char>();
        TestToLower<wchar16>();
    }
}

Y_UNIT_TEST_SUITE(TFastFindFirstOf) {
    Y_UNIT_TEST(Test0) {
        const char* s = "abcd";

        UNIT_ASSERT_EQUAL(FastFindFirstOf(s, 4, nullptr, 0) - s, 4);
    }

    Y_UNIT_TEST(Test1) {
        const char* s = "abcd";

        UNIT_ASSERT_EQUAL(FastFindFirstOf(s, 4, "b", 1) - s, 1);
    }

    Y_UNIT_TEST(Test1NotFound) {
        const char* s = "abcd";

        UNIT_ASSERT_EQUAL(FastFindFirstOf(s, 4, "x", 1) - s, 4);
    }

    Y_UNIT_TEST(Test2) {
        const char* s = "abcd";

        UNIT_ASSERT_EQUAL(FastFindFirstOf(s, 4, "xc", 2) - s, 2);
    }

    Y_UNIT_TEST(Test3) {
        const char* s = "abcde";

        UNIT_ASSERT_EQUAL(FastFindFirstOf(s, 5, "edc", 3) - s, 2);
    }

    Y_UNIT_TEST(TestNot) {
        const char* s = "abcd";

        UNIT_ASSERT_EQUAL(FastFindFirstNotOf(s, 4, "ab", 2) - s, 2);
    }
}
