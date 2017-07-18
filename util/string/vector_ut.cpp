#include <library/unittest/registar.h>
#include <util/charset/wide.h>

#include "cast.h"
#include "vector.h"

SIMPLE_UNIT_TEST_SUITE(TStringJoinTest) {
    SIMPLE_UNIT_TEST(Test1) {
        yvector<TUtf16String> v;

        UNIT_ASSERT_EQUAL(JoinStrings(v, ToWtring("")), ToWtring(""));
    }

    SIMPLE_UNIT_TEST(Test2) {
        yvector<TUtf16String> v;

        v.push_back(ToWtring("1"));
        v.push_back(ToWtring("2"));

        UNIT_ASSERT_EQUAL(JoinStrings(v, ToWtring(" ")), ToWtring("1 2"));
    }

    SIMPLE_UNIT_TEST(Test3) {
        yvector<TUtf16String> v;

        v.push_back(ToWtring("1"));
        v.push_back(ToWtring("2"));

        UNIT_ASSERT_EQUAL(JoinStrings(v, 1, 10, ToWtring(" ")), ToWtring("2"));
    }

    SIMPLE_UNIT_TEST(TestJoinWStrings) {
        const TUtf16String str = UTF8ToWide("Яндекс");
        const yvector<TUtf16String> v(1, str);

        UNIT_ASSERT_EQUAL(JoinStrings(v, TUtf16String()), str);
    }
}
